#include "TestVulkanDevice.h"
#include "common/emulatorConfig.h"
#include "common/threads.h"
#include "graphics/guest_gpu/command_processor/commandProcessor.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "kernel/memory.h"
#include "prx/libkernel/include/Api.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace Libs::Graphics {

struct VramBudgetTestAccess {
	static void SetReportedMemory(GraphicContext& graphics, uint64_t budget, uint64_t usage) {
		graphics.m_memory_override_enabled = true;
		graphics.m_memory_budget_override  = budget;
		graphics.m_memory_usage_override   = usage;
	}

	static uint64_t TextureTrigger(const TextureCache& cache) { return cache.m_trigger_gc_memory; }
	static uint64_t TexturePressure(const TextureCache& cache) { return cache.m_pressure_gc_memory; }
	static uint64_t TextureCritical(const TextureCache& cache) { return cache.m_critical_gc_memory; }
	static uint64_t BufferTrigger(const BufferCache& cache) { return cache.m_trigger_gc_memory; }
	static uint64_t BufferCritical(const BufferCache& cache) { return cache.m_critical_gc_memory; }

	static std::vector<vk::DescriptorPool> PendingDescriptorPools(const DescriptorHeap& heap) {
		std::vector<vk::DescriptorPool> pools;
		for (const auto& [pool, tick]: heap.m_pending_pools) {
			pools.push_back(pool.pool);
		}
		return pools;
	}

	static ImageId CreateImage(TextureCache& cache, const ImageInfo& info) {
		Common::ScopedLock lock {cache.m_lock};
		return cache.InsertImage(info);
	}

	static void Collect(TextureCache& cache, std::chrono::steady_clock::time_point now) {
		cache.CollectAt(now);
	}

	static bool IsImageRegistered(TextureCache& cache, ImageId id) {
		Common::ScopedLock lock {cache.m_lock};
		const auto*        image = cache.m_slot_images.try_get(id);
		return image != nullptr && image->registered;
	}
};

}

namespace {

using namespace Libs::Graphics;

constexpr uint64_t GiB          = 1024ull * 1024 * 1024;
constexpr uint64_t GuestBytes   = 0x20000;
constexpr int      CpuReadWrite = 0x03;

const char* CheckDescriptorPoolRetirement(GraphicContext& graphics) {
	MasterSemaphore timeline(graphics);
	DescriptorHeap heap(graphics, timeline);
	const vk::DescriptorSetLayoutBinding binding {0, vk::DescriptorType::eStorageBuffer, 1,
	                                               vk::ShaderStageFlagBits::eCompute};
	vk::DescriptorSetLayoutCreateInfo create {};
	create.bindingCount = 1;
	create.pBindings = &binding;
	vk::DescriptorSetLayout layout;
	if (graphics.device.createDescriptorSetLayout(&create, nullptr, &layout) != vk::Result::eSuccess) {
		return "could not create the test descriptor layout";
	}
	DescriptorHeap::RegisterLayout(layout, std::span {&binding, 1});
	constexpr uint32_t SetsPerPool = 1024;
	for (uint32_t i = 0; i < 3 * SetsPerPool + 1; i++) {
		(void)heap.Commit(layout);
	}
	const auto first = timeline.NextTick();
	for (uint32_t i = 0; i < SetsPerPool; i++) {
		(void)heap.Commit(layout);
	}
	const auto last = timeline.NextTick();
	const auto pending = VramBudgetTestAccess::PendingDescriptorPools(heap);
	heap.TrimCompletedPools();
	const auto in_flight = VramBudgetTestAccess::PendingDescriptorPools(heap);

	vk::SemaphoreSignalInfo signal {};
	signal.semaphore = timeline.Handle();
	signal.value = first;
	EXIT_IF(graphics.device.signalSemaphore(&signal) != vk::Result::eSuccess);
	heap.TrimCompletedPools();
	const auto partial = VramBudgetTestAccess::PendingDescriptorPools(heap);

	signal.value = last;
	EXIT_IF(graphics.device.signalSemaphore(&signal) != vk::Result::eSuccess);
	heap.TrimCompletedPools();
	const auto ready = VramBudgetTestAccess::PendingDescriptorPools(heap);
	graphics.device.destroyDescriptorSetLayout(layout, nullptr);

	if (pending.size() != 4 || in_flight != pending) {
		return "descriptor trimming did not preserve pools awaiting GPU completion";
	}
	if (partial.size() != 2 || partial[0] != pending[2] || partial[1] != pending[3] ||
	    ready.size() != 1 || ready[0] != pending[3]) {
		return "descriptor trimming failed to retain one completed pool and all pending pools";
	}
	return nullptr;
}

const char* CheckVramBudget(GraphicContext& graphics, TextureCache& textures, BufferCache& buffers) {
	if (const auto* failure = CheckDescriptorPoolRetirement(graphics)) {
		return failure;
	}
	graphics.AdvanceFrame(1);
	const auto tex_trigger_large  = VramBudgetTestAccess::TextureTrigger(textures);
	const auto tex_pressure_large = VramBudgetTestAccess::TexturePressure(textures);
	const auto tex_critical_large = VramBudgetTestAccess::TextureCritical(textures);
	const auto buf_trigger_large  = VramBudgetTestAccess::BufferTrigger(buffers);
	const auto buf_critical_large = VramBudgetTestAccess::BufferCritical(buffers);

	VramBudgetTestAccess::SetReportedMemory(graphics, 6 * GiB, 0);
	textures.RunGarbageCollector();
	buffers.RunGarbageCollector();

	const auto tex_trigger_small  = VramBudgetTestAccess::TextureTrigger(textures);
	const auto tex_pressure_small = VramBudgetTestAccess::TexturePressure(textures);
	const auto tex_critical_small = VramBudgetTestAccess::TextureCritical(textures);
	const auto buf_trigger_small  = VramBudgetTestAccess::BufferTrigger(buffers);
	const auto buf_critical_small = VramBudgetTestAccess::BufferCritical(buffers);

	if (tex_trigger_small >= tex_trigger_large || tex_pressure_small >= tex_pressure_large ||
	    tex_critical_small >= tex_critical_large) {
		return "texture cache GC did not refresh thresholds when reported budget decreased";
	}
	if (tex_trigger_small != 0) {
		return "texture cache trigger threshold was non-zero at <= 8 GiB budget";
	}
	if (buf_trigger_small >= buf_trigger_large || buf_critical_small >= buf_critical_large) {
		return "buffer cache GC did not refresh thresholds when reported budget decreased";
	}

	vk::ImageCreateInfo recycle_info {};
	recycle_info.imageType     = vk::ImageType::e2D;
	recycle_info.format        = vk::Format::eR8G8B8A8Unorm;
	recycle_info.extent        = vk::Extent3D {64, 64, 1};
	recycle_info.mipLevels     = 1;
	recycle_info.arrayLayers   = 1;
	recycle_info.samples       = vk::SampleCountFlagBits::e1;
	recycle_info.tiling        = vk::ImageTiling::eOptimal;
	recycle_info.usage         = vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled;
	recycle_info.sharingMode   = vk::SharingMode::eExclusive;
	recycle_info.initialLayout = vk::ImageLayout::eUndefined;

	VulkanImage recycled_image {};
	if (!graphics.CreateImage(recycle_info, recycled_image)) {
		return "could not create recyclable image for recycle pool check";
	}
	graphics.DeleteImage(recycled_image);
	if (graphics.RecycledImageCount() == 0) {
		return "deleted recyclable image did not enter the recycle pool";
	}

	VramBudgetTestAccess::SetReportedMemory(graphics, 6 * GiB, 6 * GiB);
	textures.RunGarbageCollector();
	if (graphics.RecycledImageCount() != 0) {
		return "pressure GC pass did not drain the recycled image pool";
	}
	return nullptr;
}

const char* CheckGuestEviction(GraphicContext& graphics, TextureCache& textures, uint64_t address) {
	ImageInfo info;
	info.data.address = address;
	info.pixel_format = vk::Format::eR8G8B8A8Unorm;
	info.guest_format = Prospero::BufferFormat::k8_8_8_8UNorm;
	info.extent       = vk::Extent3D {64, 64, 1};
	info.pitch        = 64;
	info.UpdateSize();

	auto now = std::chrono::steady_clock::time_point {};
	const auto check_age = [&](uint64_t usage, unsigned idle_passes) -> const char* {
		VramBudgetTestAccess::SetReportedMemory(graphics, 6 * GiB, usage);
		const auto id = VramBudgetTestAccess::CreateImage(textures, info);
		if (!VramBudgetTestAccess::IsImageRegistered(textures, id)) {
			return "test image was not registered";
		}
		for (unsigned pass = 0; pass < idle_passes; ++pass) {
			now += std::chrono::milliseconds(1);
			VramBudgetTestAccess::Collect(textures, now);
		}
		if (!VramBudgetTestAccess::IsImageRegistered(textures, id)) {
			return "texture was evicted before the idle age for its pressure band";
		}
		now += std::chrono::milliseconds(1);
		VramBudgetTestAccess::Collect(textures, now);
		if (VramBudgetTestAccess::IsImageRegistered(textures, id)) {
			return "texture was retained at the idle age for its pressure band";
		}
		return nullptr;
	};
	if (const auto* failure = check_age(3 * GiB, 80)) {
		return failure;
	}
	return check_age(6 * GiB, 160);
}

const char* CheckBudgetLimitAndSpill(GraphicContext& graphics, bool expect_spill, bool& skipped) {
	skipped = false;
	if (graphics.physical_device_properties.deviceType != vk::PhysicalDeviceType::eDiscreteGpu ||
	    graphics.spill_allocator == nullptr) {
		skipped = expect_spill;
		return nullptr;
	}
	constexpr uint64_t LargeHeap = 4ull << 30u;
	bool               has_large = false;
	for (uint32_t heap = 0; heap < graphics.physical_device_memory_properties.memoryHeapCount; ++heap) {
		const auto& info = graphics.physical_device_memory_properties.memoryHeaps[heap];
		if ((info.flags & vk::MemoryHeapFlagBits::eDeviceLocal) && info.size >= LargeHeap) {
			has_large = true;
		}
	}
	if (!has_large) {
		skipped = expect_spill;
		return nullptr;
	}
	constexpr uint64_t MiB = 1024ull * 1024;
	if (!expect_spill) {
		std::array<GraphicContext::HeapUsage, VK_MAX_MEMORY_HEAPS> heaps {};
		const auto count = graphics.GetHeapUsage(heaps);
		uint64_t   cap   = 0;
		for (uint32_t heap = 0; heap < count; ++heap) {
			if (heaps[heap].device_local) {
				const uint64_t size = graphics.physical_device_memory_properties.memoryHeaps[heap].size;
				const uint64_t reserve = size >= LargeHeap ? std::min<uint64_t>(3072ull << 20u, size / 4) : 0;
				cap += std::min<uint64_t>(heaps[heap].budget, size - reserve);
			}
		}
		if (cap != 0 && graphics.GetTotalMemoryBudget() < cap) {
			return "GetTotalMemoryBudget subtracted the eighth reserve on top of the heap limit";
		}
		return nullptr;
	}
	if (graphics.GetTotalMemoryBudget() > MiB) {
		return "KYTY_MEMORY_BUDGET_MB=1 did not cap GetTotalMemoryBudget to 1 MiB";
	}
	std::array<GraphicContext::HeapUsage, VK_MAX_MEMORY_HEAPS> before {};
	const auto count = graphics.GetHeapUsage(before);
	uint32_t   before_host_allocs = 0;
	for (uint32_t heap = 0; heap < count; ++heap) {
		if (!before[heap].device_local) {
			before_host_allocs += before[heap].allocations;
		}
	}
	{
		RenderContext renderer(graphics);
		Buffer        spilled(graphics, renderer.GetCommandScheduler(), MemoryUsage::DeviceLocal, 0, AllFlags,
		                      2 * MiB);
		if (spilled.IsDeviceLocal()) {
			return "2 MiB DeviceLocal buffer did not spill past the 1 MiB heap cap";
		}
		std::array<GraphicContext::HeapUsage, VK_MAX_MEMORY_HEAPS> after {};
		const auto after_count = graphics.GetHeapUsage(after);
		uint32_t after_host_allocs = 0;
		for (uint32_t heap = 0; heap < after_count; ++heap) {
			if (!after[heap].device_local) {
				after_host_allocs += after[heap].allocations;
			}
		}
		if (after_host_allocs <= before_host_allocs) {
			return "GetHeapUsage did not include spilled host-heap allocation";
		}
	}
	return nullptr;
}

const char* CheckVramCase(GraphicContext& graphics, uint64_t address) {
	VramBudgetTestAccess::SetReportedMemory(graphics, (address == 0 ? 16 : 6) * GiB, 0);

	RenderContext    renderer(graphics);
	CommandProcessor producer(renderer, 0);
	producer.Reset();
	auto& scheduler = renderer.GetCommandScheduler();
	auto& textures  = renderer.GetTextureCache();
	scheduler.Begin(producer.GetCtx(), producer.GetUcfg(), producer.GetShCtx());

	const auto* failure = address == 0 ? CheckVramBudget(graphics, textures, renderer.GetBufferCache())
	                                  : CheckGuestEviction(graphics, textures, address);
	const auto last = scheduler.CurrentTick();
	scheduler.Wait(last);
	scheduler.WaitPriorityOperations(last);
	return failure;
}

}

extern "C" int RunKytyVramBudgetTest(int mode) {
	const char* name = mode == 1 ? "kyty_shim_vram_budget_guest_eviction"
	                 : mode == 2 ? "kyty_shim_vram_budget_spill"
	                             : "kyty_shim_vram_budget";
	Common::InitializeThreads();
	Config::Initialize();
	Config::Load(Config::ConfigOptions {});
	KytyShimTests::TestVulkanDevice gpu;
	if (const auto* unavailable = gpu.Initialize()) {
		const bool required = std::getenv("ANYPS5_REQUIRE_VULKAN") != nullptr;
		std::fprintf(stderr, "%s: %s, no usable Vulkan device (%s)\n",
		             name, required ? "FAILED" : "skipped", unavailable);
		return required ? 1 : 77;
	}
	if (!gpu.graphics.CreateAllocator()) {
		std::fprintf(stderr, "%s: FAILED: could not create the allocator\n", name);
		return 1;
	}
	if (mode != 1) {
		bool skipped = false;
		if (const auto* failure = CheckBudgetLimitAndSpill(gpu.graphics, mode == 2, skipped)) {
			std::fprintf(stderr, "%s: FAILED: %s\n", name, failure);
			return 1;
		}
		if (skipped) {
			std::fprintf(stderr, "%s: skipped, the device reports no supported spill path "
			                     "(requires a discrete GPU, spill allocator, and 4 GiB device-local heap)\n", name);
			return 77;
		}
		if (mode == 2) {
			std::printf("%s: ok\n", name);
			return 0;
		}
	}

	uint64_t address = 0;
	if (mode == 1) {
		void* guest = nullptr;
		if (sceKernelMapFlexibleMemory(&guest, GuestBytes, CpuReadWrite, 0) != 0 || guest == nullptr) {
			std::fprintf(stderr, "%s: FAILED: could not map guest backing\n", name);
			return 1;
		}
		address = reinterpret_cast<uint64_t>(guest);
	}
	const bool invalid_range = mode == 1 && !GuestRange {address, GuestBytes}.Valid();
	const char* failure = invalid_range ? nullptr : CheckVramCase(gpu.graphics, address);
	if (mode == 1 && sceKernelMunmap(address, GuestBytes) != 0) {
		std::fprintf(stderr, "%s: FAILED: could not unmap guest backing\n", name);
		return 1;
	}
	if (invalid_range) {
		std::fprintf(stderr, "%s: skipped, guest range 0x%llx is outside Kyty bounds\n",
		             name, static_cast<unsigned long long>(address));
		return 77;
	}
	if (failure != nullptr) {
		std::fprintf(stderr, "%s: FAILED: %s\n", name, failure);
		return 1;
	}
	std::printf("%s: ok\n", name);
	return 0;
}
