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

#include <concepts>
#include <cstddef>
#include <type_traits>
#include <utility>

#include "sparrow/layout/nested_value_types.hpp"
#include "sparrow/types/data_traits.hpp"
#include "sparrow/utils/mp_utils.hpp"
#include "sparrow/utils/nullable.hpp"

namespace sparrow
{
    template <std::integral IT>
    class dictionary_encoded_array;

    /**
     * Checks whether a nullable value can be assigned to a dictionary-encoded element.
     *
     * @tparam T The nullable value type.
     * @tparam B The nullable flag type.
     */
    template <class T, class B>
    concept dictionary_assignable_nullable = std::constructible_from<array_traits::value_type, nullable<T, B>>
                                             || std::constructible_from<array_traits::const_reference, nullable<T, B>>;

    /**
     * Mutable reference to one element of a dictionary encoded array.
     *
     * Assigning through this proxy rewrites the underlying key and keeps the dictionary in sync:
     * a value already present in the dictionary reuses its entry, a new value is appended to the
     * dictionary values, and entries left without any reference are removed.
     *
     * @tparam IT The integral type used for dictionary keys.
     */
    template <std::integral IT>
    class dictionary_reference
    {
    public:

        using array_type = dictionary_encoded_array<IT>;
        using value_type = array_traits::value_type;
        using const_reference = array_traits::const_reference;

        /**
         * Constructs a reference to the element at the given index.
         *
         * @param array The referenced array.
         * @param index The index of the referenced element.
         */
        constexpr dictionary_reference(array_type* array, std::size_t index) noexcept;

        /**
         * Copy constructor.
         *
         * @param other The reference to copy.
         */
        constexpr dictionary_reference(const dictionary_reference& other) = default;

        /**
         * Move constructor.
         *
         * @param other The reference to move from.
         */
        constexpr dictionary_reference(dictionary_reference&& other) noexcept = default;

        /**
         * Assigns a value to the referenced element.
         *
         * @param value The value to assign.
         * @return Reference to this dictionary_reference.
         */
        dictionary_reference& operator=(const value_type& value);

        /**
         * Assigns a value to the referenced element.
         *
         * @param value The value to assign.
         * @return Reference to this dictionary_reference.
         */
        dictionary_reference& operator=(const const_reference& value);

        /**
         * Assigns a nullable value to the referenced element.
         *
         * @tparam T The nullable value type.
         * @tparam B The nullable flag type.
         * @param value The value to assign.
         * @return Reference to this dictionary_reference.
         */
        template <class T, mpl::boolean_like B>
            requires dictionary_assignable_nullable<T, B>
        dictionary_reference& operator=(const nullable<T, B>& value);

        /**
         * Rejects nullable rvalues.
         *
         * The nullable reference wrapper would otherwise select its generic assignment, which
         * unconditionally marks the element valid after assigning the value. Deleting the rvalue
         * overload makes it select the converting assignment instead, so the validity flag of the
         * assigned nullable is preserved.
         *
         * @tparam T The nullable value type.
         * @tparam B The nullable flag type.
         * @param value The value to assign.
         */
        template <class T, mpl::boolean_like B>
            requires dictionary_assignable_nullable<T, B>
        dictionary_reference& operator=(nullable<T, B>&& value) = delete;

        /**
         * Assigns a raw value to the referenced element.
         *
         * @tparam T The value type.
         * @param value The value to assign.
         * @return Reference to this dictionary_reference.
         */
        template <class T>
            requires(
                not std::same_as<std::decay_t<T>, dictionary_reference<IT>> and not is_nullable_v<std::decay_t<T>>
                and not std::same_as<std::decay_t<T>, typename dictionary_reference<IT>::value_type>
                and not std::same_as<std::decay_t<T>, typename dictionary_reference<IT>::const_reference>
                and not std::same_as<std::decay_t<T>, nullval_t>
                and dictionary_assignable_nullable<std::decay_t<T>, bool>
            )
        dictionary_reference& operator=(T&& value)
        {
            p_array->set_value(m_index, nullable<std::decay_t<T>>(std::forward<T>(value)));
            return *this;
        }

        /**
         * Assigns a null value to the referenced element.
         *
         * @param null_value The null sentinel.
         * @return Reference to this dictionary_reference.
         */
        dictionary_reference& operator=(nullval_t null_value);

        /**
         * Assigns the value referenced by another proxy to the referenced element.
         *
         * @param rhs The reference to copy.
         * @return Reference to this dictionary_reference.
         */
        dictionary_reference& operator=(const dictionary_reference& rhs);

        /**
         * Converts to the read-only element reference.
         *
         * @return The referenced element.
         */
        operator const_reference() const;

    private:

        /** The referenced array. */
        array_type* p_array = nullptr;
        /** The index of the referenced element. */
        std::size_t m_index = 0;
    };

    /***********************************
     * dictionary_reference implementation *
     ***********************************/

    template <std::integral IT>
    constexpr dictionary_reference<IT>::dictionary_reference(array_type* array, std::size_t index) noexcept
        : p_array(array)
        , m_index(index)
    {
    }

    template <std::integral IT>
    auto dictionary_reference<IT>::operator=(const value_type& value) -> dictionary_reference&
    {
        p_array->set_value(m_index, value);
        return *this;
    }

    template <std::integral IT>
    auto dictionary_reference<IT>::operator=(const const_reference& value) -> dictionary_reference&
    {
        return operator=(value_type(value));
    }

    template <std::integral IT>
    template <class T, mpl::boolean_like B>
        requires dictionary_assignable_nullable<T, B>
    auto dictionary_reference<IT>::operator=(const nullable<T, B>& value) -> dictionary_reference&
    {
        p_array->set_value(m_index, value);
        return *this;
    }

    template <std::integral IT>
    auto dictionary_reference<IT>::operator=(nullval_t) -> dictionary_reference&
    {
        p_array->set_value(m_index, nullval);
        return *this;
    }

    template <std::integral IT>
    auto dictionary_reference<IT>::operator=(const dictionary_reference& rhs) -> dictionary_reference&
    {
        if (this != &rhs)
        {
            const const_reference value = rhs;
            p_array->set_value(m_index, value);
        }
        return *this;
    }

    template <std::integral IT>
    dictionary_reference<IT>::operator const_reference() const
    {
        return p_array->element_at(m_index);
    }
}
