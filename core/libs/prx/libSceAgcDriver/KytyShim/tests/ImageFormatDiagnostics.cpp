#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

namespace {

using namespace Libs::Graphics;
using namespace Libs::Graphics::ShaderRecompiler;

void Require(bool condition, const char* message) {
    if (condition) return;
    std::fprintf(stdout, "kyty_shim_image_format_diagnostics: %s\n", message);
    std::exit(1);
}

IR::DescriptorValue MakeStorageDescriptor(Prospero::BufferFormat format) {
    IR::DescriptorValue value {.dword_count = 8u};
    value.dwords[0] = 0x1000u;
    value.dwords[1] = (static_cast<uint32_t>(format) << 20u) | (3u << 30u);
    value.dwords[2] = 15u;
    value.dwords[3] = 0xfacu | (9u << 28u);
    return value;
}

IR::ResourcePlan MakeAtomicImagePlan(Prospero::BufferFormat format) {
    IR::ResourcePlan plan;
    plan.stage                      = ShaderType::Compute;
    plan.resource_tracking_complete = true;
    plan.srt_plan_complete          = true;

    const auto descriptor = MakeStorageDescriptor(format);
    IR::DescriptorSource source;
    source.dword_count = 8u;
    for (size_t i = 0; i < 8u; ++i) {
        source.dwords[i] = IR::Value(descriptor.dwords[i]);
    }
    plan.descriptor_sources.push_back(source);

    IR::ImageResource image;
    image.resource_class = IR::ImageResourceClass::Storage;
    image.written        = true;
    image.atomic         = true;
    image.dimension      = Decoder::ImageDimension::Dim2D;
    image.source         = 0u;
    image.first_use_pc   = 0x1234u;
    plan.info.images.push_back(image);
    IR::CompileSrtWalker(plan);
    return plan;
}

std::string ReadFile(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}

}

int main() {
    const auto valid_plan = MakeAtomicImagePlan(Prospero::BufferFormat::k32UInt);
    IR::SrtRuntime             runtime {};
    IR::ResourceSnapshot       valid_snap;
    IR::ResourceSpecialization valid_spec;
    Require(IR::MaterializeResources(valid_plan, runtime, valid_snap, valid_spec),
            "valid 32-bit atomic image failed materialization");

    const auto log_path = std::filesystem::temp_directory_path() /
                          ("kyty-image-format-diag-" +
                           std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) +
                           ".log");
    FILE* redirected = std::freopen(log_path.string().c_str(), "w", stderr);
    Require(redirected != nullptr, "failed to redirect stderr");

    const auto                 invalid_plan = MakeAtomicImagePlan(Prospero::BufferFormat::k8UInt);
    IR::ResourceSnapshot       invalid_snap;
    IR::ResourceSpecialization invalid_spec;
    const bool ok = IR::MaterializeResources(invalid_plan, runtime, invalid_snap, invalid_spec);
    std::fflush(stderr);
    std::freopen("NUL", "w", stderr);
    const std::string log = ReadFile(log_path);
    std::error_code   ec;
    std::filesystem::remove(log_path, ec);

    Require(!ok, "8-bit atomic image unexpectedly materialized");
    if (log.find("atomic image descriptor 0 uses unsupported format 5: root=0 source=0 "
                 "first_use_pc=0x1234 r128=false words=00001000,c0500000,0000000f,90000fac,"
                 "00000000,00000000,00000000,00000000") == std::string::npos) {
        std::fprintf(stdout, "unexpected log: %s\n", log.c_str());
        return 1;
    }

    std::puts("kyty_shim_image_format_diagnostics: ok");
    return 0;
}
