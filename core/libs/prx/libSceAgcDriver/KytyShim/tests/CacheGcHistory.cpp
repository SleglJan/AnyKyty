#include "graphics/host_gpu/renderer/cache/cacheGcHistory.h"

#include <array>
#include <chrono>
#include <cstdio>
#include <optional>

template <size_t Age>
bool CheckCutoff() {
	using namespace std::chrono_literals;
	using History = Libs::Graphics::CacheGcHistory<160>;
	History history;
	std::array<History::Clock::time_point, 640> ends {};
	auto now = History::Clock::time_point {};
	for (size_t pass = 0; pass < ends.size(); ++pass) {
		auto interval = pass % 32 < 16 ? 8ms : 16ms;
		if (pass % 192 == 0) interval = 2857ms;
		now += interval;
		ends[pass] = now;
		history.FinishPass(pass, now);
		std::optional<uint64_t> expected;
		for (size_t last = 0; last <= pass; ++last) {
			if (pass - last >= Age || (pass - last >= 3 && now - ends[last] >= 16ms * Age)) {
				expected = last;
			}
		}
		if (history.Cutoff<Age>(pass) != expected) {
			std::fprintf(stderr, "incorrect idle cutoff for age %zu on pass %zu\n", Age, pass);
			return false;
		}
	}
	return true;
}

int main() {
	return CheckCutoff<16>() && CheckCutoff<80>() && CheckCutoff<160>() ? 0 : 1;
}
