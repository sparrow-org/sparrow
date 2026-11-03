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

#include "sparrow/layout/array_helper.hpp"

#include <iterator>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "sparrow/array.hpp"
#include "sparrow/layout/array_registry.hpp"
#include "sparrow/timestamp_array.hpp"
#include "sparrow/variable_size_binary_array.hpp"

namespace sparrow
{
    namespace
    {

        array_traits::value_type default_value(const array& source)
        {
            return source.visit(
                []<typename T>(const T& typed_array) -> array_traits::value_type
                {
                    using value_type = typename T::inner_value_type;
                    if constexpr (std::same_as<value_type, array_traits::inner_value_type>)
                    {
                        return std::visit(
                            []<typename U>(const U& value) -> array_traits::value_type
                            {
                                using alternative_type = U;
                                return nullable<alternative_type>(value);
                            },
                            value_type{}
                        );
                    }
                    else
                    {
                        return nullable<value_type>(value_type{});
                    }
                }
            );
        }

        template <class Value>
        array array_make_from_element_impl(Value&& value)
        {
            using value_type = std::remove_cvref_t<Value>;
            using value_base_type = typename value_type::base_type;

            auto visitor = []<typename T>(T&& typed_value) -> array
            {
                using nullable_type = std::remove_cvref_t<T>;
                using source_value_type = std::remove_cvref_t<typename nullable_type::value_type>;
                const bool has_value = typed_value.has_value();
                auto make_stored_value = [&]() -> nullable<source_value_type>
                {
                    if constexpr (std::is_lvalue_reference_v<T>)
                    {
                        return nullable<source_value_type>(source_value_type(typed_value.get()), has_value);
                    }
                    else
                    {
                        return nullable<source_value_type>(std::forward<T>(typed_value));
                    }
                };

                if constexpr (std::same_as<source_value_type, null_type>)
                {
                    return array{null_array(1)};
                }
                else if constexpr (std::same_as<source_value_type, std::string>)
                {
                    std::vector<nullable<std::string>> values;
                    values.emplace_back(make_stored_value());
                    return array{string_array(std::move(values))};
                }
                else if constexpr (std::same_as<source_value_type, std::vector<byte_t>>)
                {
                    std::vector<nullable<source_value_type>> values;
                    values.emplace_back(make_stored_value());
                    return array{binary_array(std::move(values))};
                }
                else if constexpr (std::same_as<source_value_type, list_value>
                                   || std::same_as<source_value_type, map_value>
                                   || std::same_as<source_value_type, struct_value>)
                {
                    throw std::invalid_argument("Cannot create a union child from a nested value");
                }
                else
                {
                    if constexpr (mpl::is_type_instance_of_v<source_value_type, timestamp>)
                    {
                        const auto* timezone = typed_value.get().get_time_zone();
                        std::vector<nullable<source_value_type>> values;
                        values.emplace_back(make_stored_value());
                        return array{timestamp_array<source_value_type>(timezone, std::move(values))};
                    }
                    else if constexpr (decimal_type<source_value_type>)
                    {
                        using integer_type = typename source_value_type::integer_type;
                        std::vector<integer_type> storage_values{typed_value.get().storage()};
                        std::vector<bool> validity{has_value};
                        constexpr std::size_t precision = sizeof(integer_type) == 4
                                                               ? 9
                                                               : sizeof(integer_type) == 8
                                                                     ? 18
                                                                     : sizeof(integer_type) == 16 ? 38 : 76;
                        return array(
                            decimal_array<source_value_type>(
                                std::move(storage_values),
                                std::move(validity),
                                precision,
                                typed_value.get().scale()
                            )
                        );
                    }
                    else
                    {
                        std::vector<nullable<source_value_type>> values;
                        values.emplace_back(make_stored_value());
                        return array(primitive_array_impl<source_value_type>(std::move(values)));
                    }
                }
            };

#if SPARROW_GCC_11_2_WORKAROUND
            if constexpr (std::is_lvalue_reference_v<Value>)
            {
                return std::visit(visitor, static_cast<const value_base_type&>(value));
            }
            else
            {
                return std::visit(visitor, static_cast<value_base_type&&>(std::move(value)));
            }
#else
            return std::visit(visitor, std::forward<Value>(value));
#endif
        }

        template <class ARRAY>
        constexpr bool supports_element_insert_v = is_nullable_v<typename ARRAY::value_type>
                                                   && requires(
                                                       ARRAY& array,
                                                       typename ARRAY::const_iterator pos,
                                                       const typename ARRAY::value_type& typed_value,
                                                       std::size_t count
                                                   ) { array.insert(pos, typed_value, count); };
    }

    array make_array_view(const array_wrapper& source)
    {
        return array{source.get_arrow_proxy().view()};
    }

    std::vector<array_traits::value_type> snapshot_array(const array& source)
    {
        std::vector<array_traits::value_type> result;
        result.reserve(source.size());
        for (const auto& value : source)
        {
            result.push_back(value);
        }
        return result;
    }

    void append_values(array& destination, const std::vector<array_traits::value_type>& values)
    {
        for (const auto& value : values)
        {
            array source = array_make_from_element(value);
            destination.insert(destination.cend(), source.cbegin(), source.cend());
        }
    }

    void append_values(array& destination, std::vector<array_traits::value_type>&& values)
    {
        for (auto& value : values)
        {
            array source = array_make_from_element(std::move(value));
            destination.insert(destination.cend(), source.cbegin(), source.cend());
        }
    }

    void insert_element(array& destination, std::size_t pos, const array_traits::value_type& value)
    {
        using value_variant_type = typename array_traits::value_type::base_type;

        destination.visit(
            [&](const auto& destination_impl)
            {
                using destination_array_type = std::decay_t<decltype(destination_impl)>;
                const std::string destination_format{
                    data_type_to_format(detail::get_data_type_from_array<destination_array_type>::get())
                };
                if constexpr (supports_element_insert_v<destination_array_type>)
                {
                    bool inserted = false;
                    std::visit(
                        [&](const auto& typed_value)
                        {
                            if constexpr (std::same_as<
                                              std::decay_t<decltype(typed_value)>,
                                              typename destination_array_type::value_type>)
                            {
                                auto& mutable_destination = const_cast<destination_array_type&>(destination_impl
                                );
                                auto insert_pos = std::next(
                                    mutable_destination.cbegin(),
                                    static_cast<std::ptrdiff_t>(pos)
                                );
                                mutable_destination.insert(insert_pos, typed_value, 1);
                                inserted = true;
                            }
                        },
                        static_cast<const value_variant_type&>(value)
                    );

                    if (!inserted)
                    {
                        throw std::invalid_argument(
                            "insert_element: value type does not match the destination array value "
                            "type (destination format: "
                            + destination_format + ")"
                        );
                    }
                }
                else
                {
                    throw std::runtime_error(
                        "insert_element: destination array does not support insert(value, count) "
                        "(destination format: "
                        + destination_format + ")"
                    );
                }
            }
        );
    }

    std::size_t array_size(const array_wrapper& ar)
    {
        return visit(
            [](const auto& impl)
            {
                return impl.size();
            },
            ar
        );
    }

    bool array_has_value(const array_wrapper& ar, std::size_t index)
    {
        return visit(
            [index](const auto& impl)
            {
                return impl[index].has_value();
            },
            ar
        );
    }

    array_traits::const_reference array_element(const array_wrapper& ar, std::size_t index)
    {
        using return_type = array_traits::const_reference;
        return visit(
            [index](const auto& impl) -> return_type
            {
                return return_type(impl[index]);
            },
            ar
        );
    }

    array_traits::inner_value_type array_default_element_value(const array_wrapper& ar)
    {
        using return_type = array_traits::inner_value_type;
        return visit(
            []<typename T>(const T& impl) -> return_type
            {
                using value_type = T::inner_value_type;
                return value_type();
            },
            ar
        );
    }

    array_traits::value_type array_default_value(const array_wrapper& ar)
    {
        const auto value = array_default_element_value(ar);
        return std::visit(
            []<typename T>(const T& element) -> array_traits::value_type
            {
                return nullable<T>(element);
            },
            value
        );
    }

    array array_make_from_element(const array_traits::value_type& value)
    {
        return array_make_from_element_impl(value);
    }

    array array_make_from_element(array_traits::value_type&& value)
    {
        return array_make_from_element_impl(std::move(value));
    }

    array array_empty_like(const array& source)
    {   
        return source.slice(0, 0);
    }
}
