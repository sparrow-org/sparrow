// Copyright 2024 Man Group Operations Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <concepts>
#include <cstddef>
#include <cstdint>

namespace sparrow
{
    /**
     * Checks whether the bit at the given position is set in a bitmap.
     *
     * Bits are laid out the way Apache Arrow stores validity bitmaps: the bit of index \p pos is
     * the (pos % 8)-th bit of the (pos / 8)-th byte, counted from the least significant bit.
     *
     * @param bits The bitmap.
     * @param pos The position of the bit to read.
     * @return true if the bit at \p pos is set.
     */
    [[nodiscard]] constexpr bool is_bit_set(const std::uint8_t* bits, std::size_t pos) noexcept
    {
        return (bits[pos / 8] & static_cast<std::uint8_t>(1u << (pos % 8))) != 0;
    }

    /**
     * Reverses the bytes in the given integer value.
     * \tparam T The type of the integer value.
     * \param value The integer value to reverse.
     * \return The integer value with the bytes reversed.
     * \note Implementation comes from https://en.cppreference.com/w/cpp/numeric/byteswap
     */
    template <std::integral T>
    [[nodiscard]] constexpr T byteswap(T value) noexcept
    {
        static_assert(std::has_unique_object_representations_v<T>, "T may not have padding bits");
        auto value_representation = std::bit_cast<std::array<std::byte, sizeof(T)>>(value);
        std::ranges::reverse(value_representation);
        return std::bit_cast<T>(value_representation);
    }

    template <std::endian input_value_endianess>
    [[nodiscard]] constexpr auto to_native_endian(std::integral auto value) noexcept
    {
        if constexpr (std::endian::native != input_value_endianess)
        {
            return byteswap(value);
        }
        else
        {
            return value;
        }
    }
}
