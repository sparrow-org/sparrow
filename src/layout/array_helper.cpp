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

#include <stdexcept>
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
            result.push_back(array_materialize_element(value));
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

    array_traits::value_type array_materialize_element(const array_traits::const_reference& value)
    {
        using return_type = array_traits::value_type;
        using const_reference_base_type = typename array_traits::const_reference::base_type;

        return std::visit(
            []<typename nullable_type>(const nullable_type& typed_value) -> return_type
            {
                using source_value_type = std::remove_cvref_t<typename nullable_type::value_type>;

                if constexpr (std::same_as<source_value_type, std::string_view>)
                {
                    return return_type(
                        nullable<std::string>(std::string(typed_value.get()), typed_value.has_value())
                    );
                }
                else if constexpr (std::same_as<source_value_type, sequence_view<const byte_t>>)
                {
                    const auto bytes = typed_value.get();
                    return return_type(
                        nullable<std::vector<byte_t>>(
                            std::vector<byte_t>(bytes.begin(), bytes.end()),
                            typed_value.has_value()
                        )
                    );
                }
                else
                {
                    using stored_type = std::remove_cvref_t<decltype(typed_value.get())>;
                    return return_type(
                        nullable<stored_type>(stored_type(typed_value.get()), typed_value.has_value())
                    );
                }
            },
#if SPARROW_GCC_11_2_WORKAROUND
            static_cast<const const_reference_base_type&>(value)
#else
            value
#endif
        );
    }

    array array_make_from_element(const array_traits::value_type& value)
    {
        using value_base_type = typename array_traits::value_type::base_type;

        return std::visit(
            []<typename T>(const T& typed_value) -> array
            {
                using nullable_type = T;
                using source_value_type = std::remove_cvref_t<typename nullable_type::value_type>;

                if constexpr (std::same_as<source_value_type, null_type>)
                {
                    return array{null_array(1)};
                }
                else if constexpr (std::same_as<source_value_type, std::string>)
                {
                    std::vector<nullable<std::string>> values;
                    values.emplace_back(
                        std::string(typed_value.get()),
                        typed_value.has_value()
                    );
                    return array{string_array(std::move(values))};
                }
                else if constexpr (std::same_as<source_value_type, std::vector<byte_t>>)
                {
                    using value_type = std::vector<byte_t>;
                    const auto& bytes = typed_value.get();
                    std::vector<nullable<value_type>> values;
                    values.emplace_back(
                        value_type(bytes.begin(), bytes.end()),
                        typed_value.has_value()
                    );
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
                    std::vector<nullable<source_value_type>> values;
                    values.emplace_back(
                        source_value_type(typed_value.get()),
                        typed_value.has_value()
                    );
                    if constexpr (mpl::is_type_instance_of_v<source_value_type, timestamp>)
                    {
                        return array{
                            timestamp_array<source_value_type>(typed_value.get().get_time_zone(), std::move(values))
                        };
                    }
                    else if constexpr (decimal_type<source_value_type>)
                    {
                        using integer_type = typename source_value_type::integer_type;
                        std::vector<integer_type> storage_values{typed_value.get().storage()};
                        std::vector<bool> validity{typed_value.has_value()};
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
                        return array(primitive_array_impl<source_value_type>(std::move(values)));
                    }
                }
            },
#if SPARROW_GCC_11_2_WORKAROUND
            static_cast<const value_base_type&>(value)
#else
            value
#endif
        );
    }

    array array_empty_like(const array& source)
    {   
        return source.slice(0, 0);
    }
}
