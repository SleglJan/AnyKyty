#include "prx/libSceAgcDriver/Execution/include/FrameCapture.hpp"

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <iterator>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::uint64_t GuestBase = 0x201800000;
constexpr std::uint64_t Block = 0x10000;
constexpr std::uint64_t Commands = GuestBase + 0x3000;
constexpr std::uint64_t Input = GuestBase + Block;
constexpr std::uint64_t Output = GuestBase + 2 * Block;
constexpr std::uint64_t Display = GuestBase + 3 * Block;
constexpr std::uint64_t GuestBytes = 4 * Block;
constexpr std::uint32_t Lanes = 32;
constexpr std::uint32_t LaneBytes = 16;
constexpr std::uint32_t Format32UInt = 20;
constexpr std::uint32_t Fill = 0xdeadbeef;

struct FormatCase {
    const char* name;
    std::uint32_t format;
    std::uint32_t load;
    std::uint32_t store;
    std::uint64_t header;
    std::uint64_t code;
    std::uint64_t source;
    std::uint64_t raw;
    std::uint64_t typed;
    std::array<std::uint32_t, 4> masks;
};

constexpr std::array Cases {
    FormatCase {"2_10_10_10", 44, 0xe9631000u, 0xe9671000u,
                GuestBase + 0x100, GuestBase + 0x800, Input, Output, Output + 0x2000,
                {0x00000003u, 0x00000ffcu, 0x003ff000u, 0xffc00000u}},
    FormatCase {"10_10_10_2", 50, 0xe9931000u, 0xe9971000u,
                GuestBase + 0x1100, GuestBase + 0x1800, Input + 0x1000, Output + 0x1000, Output + 0x3000,
                {0x000003ffu, 0x000ffc00u, 0x3ff00000u, 0xc0000000u}},
};

using Sharp = std::array<std::uint32_t, 4>;

std::uint32_t low(std::uint64_t address) {
    return static_cast<std::uint32_t>(address >> 8u);
}

std::uint32_t high(std::uint64_t address) {
    return static_cast<std::uint32_t>(address >> 40u);
}

Sharp sharp(std::uint64_t address, std::uint32_t format) {
    return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu),
            Lanes * LaneBytes, (0x01016facu & ~(0x7fu << 12u)) | (format << 12u) | (3u << 28u)};
}

std::array<std::uint32_t, 15> shader(const FormatCase& test) {
    return {0x34020084u, test.load, 0x80000201u, 0xbf8c0000u,
            0xe0701000u, 0x80010201u, 0xe0701004u, 0x80010301u,
            0xe0701008u, 0x80010401u, 0xe070100cu, 0x80010501u,
            test.store, 0x80020201u, 0xbf810000u};
}

void append(std::vector<std::uint32_t>& words, std::uint32_t opcode, std::initializer_list<std::uint32_t> payload) {
    words.push_back(0xc0000000u | (static_cast<std::uint32_t>(payload.size() - 1) << 16u) | (opcode << 8u));
    words.insert(words.end(), payload);
}

std::vector<std::uint32_t> dispatchWords() {
    std::vector<std::uint32_t> words;
    for (const auto& test: Cases) {
        const auto source = sharp(test.source, test.format);
        const auto raw = sharp(test.raw, Format32UInt);
        const auto typed = sharp(test.typed, test.format);
        append(words, 0x76, {0x20c, low(test.code), high(test.code)});
        append(words, 0x76, {0x212, 0u});
        append(words, 0x76, {0x213, 12u << 1u});
        append(words, 0x76, {0x207, Lanes, 1u, 1u});
        append(words, 0x76, {0x240, source[0], source[1], source[2], source[3],
                             raw[0], raw[1], raw[2], raw[3], typed[0], typed[1], typed[2], typed[3]});
        append(words, 0x15, {1u, 1u, 1u, 0x41u});
    }
    return words;
}

void write(std::span<std::byte> memory, std::uint64_t address, const void* data, std::size_t bytes) {
    const auto offset = address - GuestBase;
    if (offset > memory.size() || bytes > memory.size() - offset) {
        throw std::runtime_error("fixture memory write past the guest mapping");
    }
    std::memcpy(memory.data() + offset, data, bytes);
}

void writeShader(std::span<std::byte> memory, const FormatCase& test) {
    struct Header {
        Shader shader{};
        ShaderUserData user{};
    };
    const auto words = shader(test);
    Header header{};
    header.shader.file_header = 0x34333231;
    header.shader.version = 0x18;
    header.shader.header_size = sizeof(Header);
    header.shader.shader_size = static_cast<std::uint32_t>(sizeof(words));
    header.shader.code = reinterpret_cast<const void*>(test.code);
    header.shader.type = 0;
    header.shader.user_data = reinterpret_cast<ShaderUserData*>(test.header + offsetof(Header, user));
    write(memory, test.header, &header, sizeof(header));
    write(memory, test.code, words.data(), sizeof(words));
}

void makeCapture(const std::filesystem::path& path) {
    using AgcDriver::FrameCallKind;
    std::vector<std::byte> memory(GuestBytes);
    for (const auto& test: Cases) {
        writeShader(memory, test);
        std::array<std::uint32_t, Lanes * 4> input{};
        for (std::uint32_t lane = 0; lane < Lanes; ++lane) {
            input[lane * 4u] = test.masks[lane % 4u];
        }
        write(memory, test.source, input.data(), sizeof(input));
    }
    const auto commands = dispatchWords();
    write(memory, Commands, commands.data(), commands.size() * sizeof(std::uint32_t));
    std::vector<std::uint32_t> fill(Block / sizeof(std::uint32_t), Fill);
    write(memory, Output, fill.data(), Block);

    AgcDriver::FrameCaptureWriter writer(path);
    writer.WriteRange(GuestBase, memory.size(), AgcDriver::FrameRangeKind::Watched, 0, 7);
    const auto snapshot = writer.WriteSnapshot(GuestBase, memory);
    std::uint64_t ordinal = 0;
    const auto emit = [&](AgcDriver::FrameCall call) {
        call.ordinal = ++ordinal;
        call.timeNs = ordinal * 1000u;
        call.thread = 1;
        writer.WriteCall(call);
    };
    for (const auto& test: Cases) {
        emit({.kind = FrameCallKind::RegisterShader, .address = test.header});
    }
    emit({.kind = FrameCallKind::RegisterVideoOutput, .values = {7}});
    emit({.kind = FrameCallKind::DisplayBuffer, .address = Display,
          .values = {0, 0x8000000022000000ull, 64, 4, 0, 0, 0, 0, 0}});
    emit({.kind = FrameCallKind::FrameStart, .values = {1}});
    emit({.kind = FrameCallKind::Submit, .address = Commands, .size = commands.size()});
    emit({.kind = FrameCallKind::Flip, .accepted = 1, .cpuFlip = true, .values = {7, UINT64_MAX, 1, 0}});
    emit({.kind = FrameCallKind::End, .accepted = 2});
    writer.Finish("{\"magic\":\"aps5-frame-capture 1\",\"arena_base\":65536,\"arena_bytes\":1082331693056,"
                  "\"start_boundary\":1,\"frames\":1,\"calls\":2,\"prologue_calls\":4,\"ranges\":1,"
                  "\"snapshot_bytes\":" + std::to_string(snapshot) +
                  ",\"unreadable_bytes\":0,\"host_only_sample_bytes\":0,\"warnings\":[]}\n");
}

std::string readFile(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}

std::string quote(const std::filesystem::path& path) {
    return "\"" + path.string() + "\"";
}

std::string hex(std::uint32_t value) {
    char text[16];
    std::snprintf(text, sizeof(text), "0x%08x", value);
    return text;
}

void printEvidence(const std::string& log) {
    std::istringstream lines(log);
    std::string line;
    while (std::getline(lines, line)) {
        if (line.find("Select device:") != std::string::npos ||
            line.find("Selected GPU name=") != std::string::npos ||
            line.find("Insert instance layer") != std::string::npos ||
            line.find("was not found but was requested") != std::string::npos ||
            line.find("VUID-") != std::string::npos) {
            std::cout << line << '\n';
        }
    }
}

}

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("usage: kyty_shim_packed_buffer_format_tests <agc_frame_replay>");
        const std::filesystem::path tool = argv[1];
        const auto root = std::filesystem::temp_directory_path() /
                          ("kyty-packed-buffer-format-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directories(root);
        makeCapture(root / "capture");
        char address[32];
        std::snprintf(address, sizeof(address), "0x%llx", static_cast<unsigned long long>(Output));
        const auto log = root / "replay.log";
        const auto dump = root / "output.bin";
        const auto command = quote(tool) + " " + quote(root / "capture") +
                             " --strict --no-synthesize --repeat 1 --cold-pipelines --frames " + quote(root / "frames") +
                             " --dump " + address + ":" + std::to_string(Block) + ":" + quote(dump) +
                             " > " + quote(log) + " 2>&1";
        std::fflush(nullptr);
#ifdef _WIN32
        const int status = std::system(('"' + command + '"').c_str());
#else
        const int status = std::system(command.c_str());
#endif
        const auto raw = readFile(dump);
        const auto replayLog = readFile(log);
        const bool vulkanRequired = std::getenv("ANYPS5_REQUIRE_VULKAN") != nullptr;
        if (status != 0 || raw.size() != Block) {
            std::cerr << "replay did not complete" << (vulkanRequired ? "" : "; skipping (no usable Vulkan device?)") << '\n';
            std::cerr << replayLog.substr(replayLog.size() > 4000 ? replayLog.size() - 4000 : 0) << '\n';
            return vulkanRequired ? 1 : 77;
        }
        printEvidence(replayLog);
        const char* requestedLayers = std::getenv("VK_INSTANCE_LAYERS");
        const bool validationRequested = requestedLayers != nullptr &&
                                         std::strstr(requestedLayers, "VK_LAYER_KHRONOS_validation") != nullptr;
        if (vulkanRequired &&
            (replayLog.find("VUID-") != std::string::npos ||
             replayLog.find("was not found but was requested by env var VK_INSTANCE_LAYERS") != std::string::npos ||
             (validationRequested && replayLog.find("Insert instance layer \"VK_LAYER_KHRONOS_validation\"") ==
                                         std::string::npos))) {
            std::cerr << "replay failed Vulkan validation log checks\n";
            return 1;
        }
        std::vector<std::uint32_t> output(raw.size() / sizeof(std::uint32_t));
        std::memcpy(output.data(), raw.data(), raw.size());
        int failures = 0;
        for (const auto& test: Cases) {
            const auto rawBase = static_cast<std::size_t>((test.raw - Output) / sizeof(std::uint32_t));
            const auto typedBase = static_cast<std::size_t>((test.typed - Output) / sizeof(std::uint32_t));
            for (std::uint32_t lane = 0; lane < Lanes; ++lane) {
                for (std::uint32_t component = 0; component < 4u; ++component) {
                    const auto got = output[rawBase + lane * 4u + component];
                    const auto want = lane % 4u == component ? 0x3f800000u : 0u;
                    if (got != want && ++failures <= 16) {
                        std::cerr << test.name << " lane " << lane << " component " << component << ": "
                                  << hex(got) << ", expected " << hex(want) << '\n';
                    }
                }
                const auto got = output[typedBase + lane * 4u];
                const auto want = test.masks[lane % 4u];
                if (got != want && ++failures <= 16) {
                    std::cerr << test.name << " lane " << lane << " typed store: " << hex(got)
                              << ", expected " << hex(want) << '\n';
                }
            }
        }
        if (failures != 0) {
            std::cerr << failures << " wrong values\n";
            return 1;
        }
        std::filesystem::remove_all(root);
        std::cout << "kyty_shim_packed_buffer_format: ok\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
