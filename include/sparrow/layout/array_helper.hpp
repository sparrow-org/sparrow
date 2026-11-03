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
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or mplied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <vector>

#include "sparrow/config/config.hpp"
#include "sparrow/layout/array_wrapper.hpp"
#include "sparrow/types/data_traits.hpp"

// File containing helper functions to avoid bloating the
// include chain.

namespace sparrow
{
    class array;

    [[nodiscard]] SPARROW_API std::size_t array_size(const array_wrapper& ar);

    [[nodiscard]] SPARROW_API bool array_has_value(const array_wrapper& ar, std::size_t index);

    [[nodiscard]] SPARROW_API array_traits::const_reference
    array_element(const array_wrapper& ar, std::size_t index);

    [[nodiscard]] SPARROW_API array_traits::inner_value_type
    array_default_element_value(const array_wrapper& ar);

    [[nodiscard]] SPARROW_API array_traits::value_type
    array_default_value(const array_wrapper& ar);

    [[nodiscard]] SPARROW_API array
    make_array_view(const array_wrapper& source);

    [[nodiscard]] SPARROW_API std::vector<array_traits::value_type>
    snapshot_array(const array& source);

    SPARROW_API void
    append_values(array& destination, const std::vector<array_traits::value_type>& values);

    SPARROW_API void
    append_values(array& destination, std::vector<array_traits::value_type>&& values);

    /**
     * Inserts one type-erased value at the given position in a dynamically typed array.
     *
     * The value type must exactly match the destination array value type.
     *
     * @param destination The array to insert into.
     * @param pos The position before which the value is inserted.
     * @param value The value to insert.
     *
     * @throws std::invalid_argument If the value type does not match the destination value type.
     * @throws std::runtime_error If the destination does not support insertion.
     */
    SPARROW_API void insert_element(array& destination, std::size_t pos, const array_traits::value_type& value);

    [[nodiscard]] SPARROW_API array
    array_make_from_element(const array_traits::value_type& value);

    [[nodiscard]] SPARROW_API array
    array_make_from_element(array_traits::value_type&& value);

    [[nodiscard]] SPARROW_API array
    array_empty_like(const array& source);
}
