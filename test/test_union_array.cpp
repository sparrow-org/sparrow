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

#include <optional>

#include "sparrow/array.hpp"
#include "sparrow/decimal_array.hpp"
#include "sparrow/layout/array_registry.hpp"
#include "sparrow/list_array.hpp"
#include "sparrow/primitive_array.hpp"
#include "sparrow/timestamp_array.hpp"
#include "sparrow/union_array.hpp"
#include "sparrow/utils/nullable.hpp"
#include "sparrow/variable_size_binary_array.hpp"

#include "../test/external_array_data_creation.hpp"
#include "doctest/doctest.h"
#include "test_utils.hpp"

namespace sparrow
{

    namespace test
    {
        arrow_proxy
        make_sparse_union_proxy(const std::string& format_string, std::size_t n, bool altered = false)
        {
            std::vector<ArrowArray> children_arrays(2);
            std::vector<ArrowSchema> children_schemas(2);

            test::fill_schema_and_array<float32_t>(children_schemas[0], children_arrays[0], n, 0 /*offset*/, {});
            children_schemas[0].name = "item 0";

            test::fill_schema_and_array<std::uint16_t>(children_schemas[1], children_arrays[1], n, 0 /*offset*/, {});
            children_schemas[1].name = "item 1";

            ArrowArray arr{};
            ArrowSchema schema{};

            std::vector<std::uint8_t> type_ids =
                {std::uint8_t(3), std::uint8_t(4), std::uint8_t(3), std::uint8_t(4)};
            if (altered)
            {
                type_ids[0] = std::uint8_t(4);
            }

            test::fill_schema_and_array_for_sparse_union(
                schema,
                arr,
                std::move(children_schemas),
                std::move(children_arrays),
                type_ids,
                format_string
            );

            return arrow_proxy(std::move(arr), std::move(schema));
        }

        arrow_proxy
        make_dense_union_proxy(const std::string& format_string, std::size_t n_c, bool altered = false)
        {
            std::vector<ArrowArray> children_arrays(2);
            std::vector<ArrowSchema> children_schemas(2);

            test::fill_schema_and_array<float32_t>(children_schemas[0], children_arrays[0], n_c, 0 /*offset*/, {});
            children_schemas[0].name = "item 0";

            test::fill_schema_and_array<std::uint16_t>(
                children_schemas[1],
                children_arrays[1],
                n_c,
                0 /*offset*/,
                {}
            );
            children_schemas[1].name = "item 1";

            ArrowArray arr{};
            ArrowSchema schema{};

            std::vector<std::uint8_t> type_ids =
                {std::uint8_t(3), std::uint8_t(4), std::uint8_t(3), std::uint8_t(4)};
            if (altered)
            {
                type_ids[0] = std::uint8_t(4);
            }
            std::vector<std::int32_t> offsets = {0, 0, 1, 1};

            test::fill_schema_and_array_for_dense_union(
                schema,
                arr,
                std::move(children_schemas),
                std::move(children_arrays),
                type_ids,
                offsets,
                format_string
            );

            return arrow_proxy(std::move(arr), std::move(schema));
        }
    }

    TEST_SUITE("sparse_union")
    {
        static_assert(is_sparse_union_array_v<sparse_union_array>);
        static_assert(!is_dense_union_array_v<sparse_union_array>);

        TEST_CASE("constructor")
        {
            // the child arrays
            primitive_array<std::int16_t> arr1({{std::int16_t(2), std::int16_t(5), std::size_t(9)}});
            primitive_array<std::int32_t> arr2(
                std::vector<std::int32_t>{std::int32_t(3), std::int32_t(4), std::size_t(5)},
                std::vector<std::size_t>{1}  // INDEX 1 IS MISSING
            );

            // detyped arrays
            std::vector<array> children = {array(std::move(arr1)), array(std::move(arr2))};


            SUBCASE("with mapping")
            {
                // type ids
                sparse_union_array::type_id_buffer_type type_ids{
                    {std::uint8_t(2), std::uint8_t(3), std::uint8_t(3)}
                };

                // mapping
                std::vector<std::size_t> type_mapping{2, 3};

                // the array
                sparse_union_array arr(
                    std::move(children),
                    std::move(type_ids),
                    std::make_optional(std::move(type_mapping))
                );

                // check the size
                REQUIRE_EQ(arr.size(), 3);

                // check elements have values
                CHECK(arr[0].has_value());
                CHECK(!arr[1].has_value());
                CHECK(arr[2].has_value());

                CHECK_NULLABLE_VARIANT_EQ(arr[0], std::int16_t(2));
                CHECK_NULLABLE_VARIANT_EQ(arr[2], std::int32_t(5));
            }
            SUBCASE("without mapping")
            {
                // type ids
                sparse_union_array::type_id_buffer_type type_ids{
                    {std::uint8_t(0), std::uint8_t(1), std::uint8_t(1)}
                };

                // the array
                sparse_union_array arr(std::move(children), std::move(type_ids));

                // check the size
                REQUIRE_EQ(arr.size(), 3);

                // check elements have values
                CHECK(arr[0].has_value());
                CHECK(!arr[1].has_value());
                CHECK(arr[2].has_value());

                CHECK_NULLABLE_VARIANT_EQ(arr[0], std::int16_t(2));
                CHECK_NULLABLE_VARIANT_EQ(arr[2], std::int32_t(5));
            }
        }

        TEST_CASE("basics")
        {
            const std::string format_string = "+us:3,4";
            const std::size_t n = 4;


            auto proxy = test::make_sparse_union_proxy(format_string, n);
            sparse_union_array uarr(std::move(proxy));

            REQUIRE(uarr.size() == n);

            SUBCASE("copy")
            {
#ifdef SPARROW_TRACK_COPIES
                copy_tracker::reset(copy_tracker::key<sparse_union_array>());
#endif
                sparse_union_array uarr2(uarr);
                CHECK_EQ(uarr2, uarr);
#ifdef SPARROW_TRACK_COPIES
                CHECK_EQ(copy_tracker::count(copy_tracker::key<sparse_union_array>()), 1);
#endif

                sparse_union_array uarr3(test::make_sparse_union_proxy(format_string, n, true));
                CHECK_NE(uarr3, uarr);
                uarr3 = uarr;
                CHECK_EQ(uarr3, uarr);
            }

            SUBCASE("move")
            {
                sparse_union_array uarr2(uarr);
                sparse_union_array uarr3(std::move(uarr2));
                CHECK_EQ(uarr3, uarr);

                sparse_union_array uarr4(test::make_sparse_union_proxy(format_string, n, true));
                CHECK_NE(uarr4, uarr);
                uarr4 = std::move(uarr3);
                CHECK_EQ(uarr4, uarr);
            }

            SUBCASE("operator[]")
            {
                for (std::size_t i = 0; i < n; ++i)
                {
                    const auto& val = uarr[i];
                    REQUIRE(val.has_value());
                }

#if SPARROW_GCC_11_2_WORKAROUND
                using variant_type = std::decay_t<decltype(uarr[0])>;
                using base_type = typename variant_type::base_type;
#endif
                // 0
                std::visit(
                    [](auto&& arg)
                    {
                        using inner_type = std::decay_t<typename std::decay_t<decltype(arg)>::value_type>;
                        if constexpr (std::is_same_v<inner_type, float32_t>)
                        {
                            REQUIRE_EQ(0.0f, arg.value());
                        }
                        else
                        {
                            CHECK(false);
                        }
                    },

#if SPARROW_GCC_11_2_WORKAROUND
                    static_cast<const base_type&>(uarr[0])
#else
                    uarr[0]
#endif
                );

                // 1
                std::visit(
                    [](auto&& arg)
                    {
                        using inner_type = std::decay_t<typename std::decay_t<decltype(arg)>::value_type>;
                        if constexpr (std::is_same_v<inner_type, std::uint16_t>)
                        {
                            REQUIRE_EQ(1, arg.value());
                        }
                        else
                        {
                            CHECK(false);
                        }
                    },
#if SPARROW_GCC_11_2_WORKAROUND
                    static_cast<const base_type&>(uarr[1])
#else
                    uarr[1]
#endif
                );

                // 2
                std::visit(
                    [](auto&& arg)
                    {
                        using inner_type = std::decay_t<typename std::decay_t<decltype(arg)>::value_type>;
                        if constexpr (std::is_same_v<inner_type, float32_t>)
                        {
                            REQUIRE_EQ(2.0f, arg.value());
                        }
                        else
                        {
                            CHECK(false);
                        }
                    },
#if SPARROW_GCC_11_2_WORKAROUND
                    static_cast<const base_type&>(uarr[2])
#else
                    uarr[2]
#endif
                );

                // 3
                std::visit(
                    [](auto&& arg)
                    {
                        using inner_type = std::decay_t<typename std::decay_t<decltype(arg)>::value_type>;
                        if constexpr (std::is_same_v<inner_type, std::uint16_t>)
                        {
                            REQUIRE_EQ(3, arg.value());
                        }
                        else
                        {
                            CHECK(false);
                        }
                    },
#if SPARROW_GCC_11_2_WORKAROUND
                    static_cast<const base_type&>(uarr[3])
#else
                    uarr[3]
#endif
                );
            }
        }

        TEST_CASE("mutation")
        {
            auto proxy = test::make_sparse_union_proxy("+us:3,4", 4);
            sparse_union_array uarr(std::move(proxy));

            uarr.insert(
                uarr.cbegin() + 1,
                array_traits::value_type{make_nullable(std::int32_t(42))}
            );

            REQUIRE_EQ(uarr.size(), 5);
            CHECK_NULLABLE_VARIANT_EQ(uarr[1], std::int32_t(42));
            CHECK_EQ(detail::array_access::get_arrow_proxy(uarr).children().size(), 3);
            CHECK_EQ(detail::array_access::get_arrow_proxy(uarr).format(), "+us:3,4,0");
            for (const auto& child : detail::array_access::get_arrow_proxy(uarr).children())
            {
                CHECK_EQ(child.length(), 5);
            }

            uarr.erase(uarr.cbegin() + 1);
            CHECK_EQ(uarr.size(), 4);
            CHECK_NULLABLE_VARIANT_EQ(uarr[1], std::uint16_t(1));

            std::vector<array_traits::value_type> inserted{
                make_nullable(std::int32_t(7)),
                make_nullable(std::int32_t(8))
            };
            uarr.insert(uarr.cend(), inserted.cbegin(), inserted.cend());
            CHECK_EQ(uarr.size(), 6);
            CHECK_NULLABLE_VARIANT_EQ(uarr[4], std::int32_t(7));
            CHECK_NULLABLE_VARIANT_EQ(uarr[5], std::int32_t(8));

            uarr.resize(8, array_traits::value_type{make_nullable(std::int32_t(9))});
            CHECK_EQ(uarr.size(), 8);
            CHECK_NULLABLE_VARIANT_EQ(uarr[6], std::int32_t(9));
            CHECK_NULLABLE_VARIANT_EQ(uarr[7], std::int32_t(9));
            uarr.resize(4);
            CHECK_EQ(uarr.size(), 4);

            uarr.resize(5);
            CHECK_EQ(uarr.size(), 5);
            CHECK(!uarr[4].has_value());
            CHECK_EQ(detail::array_access::get_arrow_proxy(uarr).children().size(), 4);
            CHECK_EQ(detail::array_access::get_arrow_proxy(uarr).format(), "+us:3,4,0,1");
            for (const auto& child : detail::array_access::get_arrow_proxy(uarr).children())
            {
                CHECK_EQ(child.length(), 5);
            }
        }

        TEST_CASE("mutation edge cases")
        {
            SUBCASE("zero-count and empty-range insertion are no-ops")
            {
                sparse_union_array uarr(test::make_sparse_union_proxy("+us:3,4", 4));
                std::vector<array_traits::value_type> empty_values;

                uarr.insert(uarr.cbegin() + 1, uarr[0], 0);
                uarr.insert(uarr.cbegin() + 1, empty_values.cbegin(), empty_values.cend());

                CHECK_EQ(uarr.size(), 4);
                CHECK_EQ(detail::array_access::get_arrow_proxy(uarr).format(), "+us:3,4");
            }

            SUBCASE("caches repeated new alternatives")
            {
                sparse_union_array uarr(test::make_sparse_union_proxy("+us:3,4", 4));
                std::vector<array_traits::value_type> values{
                    make_nullable(std::int32_t(7)),
                    make_nullable(std::int32_t(8))
                };

                uarr.insert(uarr.cend(), values.cbegin(), values.cend());

                CHECK_EQ(detail::array_access::get_arrow_proxy(uarr).format(), "+us:3,4,0");
                CHECK_EQ(detail::array_access::get_arrow_proxy(uarr).children().size(), 3);
                CHECK_EQ(detail::array_access::get_arrow_proxy(uarr).children()[2].length(), 6);
                CHECK_NULLABLE_VARIANT_EQ(uarr[4], std::int32_t(7));
                CHECK_NULLABLE_VARIANT_EQ(uarr[5], std::int32_t(8));
            }

            SUBCASE("matches an existing child type")
            {
                sparse_union_array uarr(test::make_sparse_union_proxy("+us:3,4", 4));

                uarr.insert(uarr.cend(), array_traits::value_type{make_nullable(float32_t(9.0f))});

                CHECK_EQ(detail::array_access::get_arrow_proxy(uarr).format(), "+us:3,4");
                CHECK_EQ(detail::array_access::get_arrow_proxy(uarr).children().size(), 2);
                CHECK_NULLABLE_VARIANT_EQ(uarr.back(), float32_t(9.0f));
            }

            SUBCASE("supports borrowed values and self-insertion")
            {
                sparse_union_array uarr(test::make_sparse_union_proxy("+us:3,4", 4));
                uarr.insert(uarr.cend(), uarr[0]);

                array dynamic(std::move(uarr));
                dynamic.insert(dynamic.cbegin() + 1, dynamic.cbegin(), dynamic.cbegin() + 1);

                CHECK_EQ(dynamic.size(), 6);
                CHECK_NULLABLE_VARIANT_EQ(dynamic[1], float32_t(0.0f));
            }

            SUBCASE("fills nulls from empty existing children")
            {
                std::vector<array> children{
                    array(primitive_array<std::int32_t>{std::vector<std::int32_t>{}}),
                    array(string_array{std::vector<std::string>{}})
                };
                sparse_union_array uarr(
                    std::move(children),
                    sparse_union_array::type_id_buffer_type{std::vector<std::uint8_t>{}}
                );

                uarr.insert(uarr.cend(), array_traits::value_type{make_nullable(float32_t(1.0f))});

                CHECK_EQ(uarr.size(), 1);
                CHECK_EQ(detail::array_access::get_arrow_proxy(uarr).format(), "+us:0,1,2");
                CHECK_NULLABLE_VARIANT_EQ(uarr[0], float32_t(1.0f));
            }

            SUBCASE("rebuilds a bitmap-less nested union child")
            {
                auto nested = dense_union_array(
                    std::vector<array>{
                        array(primitive_array<std::int32_t>{std::vector<std::int32_t>{10}}),
                        array(string_array{std::vector<std::string>{"inner"}})
                    },
                    dense_union_array::type_id_buffer_type{std::vector<std::uint8_t>{0, 1}},
                    dense_union_array::offset_buffer_type{std::vector<std::uint32_t>{0, 0}}
                );
                sparse_union_array uarr(
                    std::vector<array>{
                        array(std::move(nested)),
                        array(primitive_array<std::int32_t>{std::vector<std::int32_t>{1, 2}})
                    },
                    sparse_union_array::type_id_buffer_type{std::vector<std::uint8_t>{0, 1}}
                );

                uarr.erase(uarr.cbegin());

                const auto& proxy = detail::array_access::get_arrow_proxy(uarr);
                REQUIRE_EQ(uarr.size(), 1);
                CHECK_NULLABLE_VARIANT_EQ(uarr[0], std::int32_t(2));
                CHECK_EQ(proxy.children()[0].length(), 1);
                CHECK_EQ(proxy.children()[0].format(), "+ud:0,1");
            }
        }

        TEST_CASE("rebuilds nested and noncanonical children")
        {
            auto list_child = list_array(
                array(primitive_array<std::int16_t>{std::vector<std::int16_t>{1, 2, 3}}),
                std::vector<std::int32_t>{0, 1, 3},
                true
            );
            auto large_string_child = big_string_array(std::vector<std::string>{"alpha", "beta"}, true);

            sparse_union_array uarr(
                std::vector<array>{array(std::move(list_child)), array(std::move(large_string_child))},
                sparse_union_array::type_id_buffer_type{std::vector<std::uint8_t>{0, 1}}
            );

            uarr.erase(uarr.cbegin());

            const auto& proxy = detail::array_access::get_arrow_proxy(uarr);
            CHECK_EQ(proxy.children()[0].format(), "+l");
            CHECK_EQ(proxy.children()[1].format(), "U");
            CHECK_EQ(proxy.children()[0].length(), 1);
            CHECK_EQ(proxy.children()[1].length(), 1);
            CHECK(uarr[0].has_value());
            array large_string_result{proxy.children()[1].view()};
            CHECK_NULLABLE_VARIANT_EQ(large_string_result[0], std::string_view("beta"));
        }

#if defined(__cpp_lib_format)
        TEST_CASE("formatting")
        {
            const std::string format_string = "+us:3,4";
            const std::size_t n = 4;

            auto proxy = test::make_sparse_union_proxy(format_string, n);
            sparse_union_array uarr(std::move(proxy));

            const std::string formatted = std::format("{}", uarr);
            constexpr std::string_view expected = "SparseUnion [name=test | size=4] <0, 1, 2, 3>";
            CHECK_EQ(formatted, expected);
        }
#endif
    }

    TEST_SUITE("dense_union")
    {
        static_assert(is_dense_union_array_v<dense_union_array>);
        static_assert(!is_sparse_union_array_v<dense_union_array>);
        TEST_CASE("constructor")
        {
            // the child arrays
            primitive_array<std::int16_t> arr1({{std::int16_t(0), std::int16_t(1)}});
            primitive_array<std::int32_t> arr2(
                std::vector<std::int32_t>{std::int32_t(2), std::int32_t(3)},
                std::vector<std::size_t>{1}  // INDEX 1 IS MISSING
            );

            // detyped arrays
            std::vector<array> children = {array(std::move(arr1)), array(std::move(arr2))};

            // offsets
            dense_union_array::offset_buffer_type offsets{
                {std::size_t(1), std::size_t(1), std::size_t(0), std::size_t(0)}
            };

            SUBCASE("without mapping")
            {
                // type ids
                dense_union_array::type_id_buffer_type type_ids{
                    {std::uint8_t(0), std::uint8_t(1), std::uint8_t(0), std::uint8_t(1)}
                };

                // the array
                dense_union_array arr(std::move(children), std::move(type_ids), std::move(offsets));

                // check the size
                REQUIRE_EQ(arr.size(), 4);

                // check elements have values
                CHECK(arr[0].has_value());
                CHECK(!arr[1].has_value());
                CHECK(arr[2].has_value());
                CHECK(arr[3].has_value());

                CHECK_NULLABLE_VARIANT_EQ(arr[0], std::int16_t(1));
                CHECK_NULLABLE_VARIANT_EQ(arr[2], std::int16_t(0));
                CHECK_NULLABLE_VARIANT_EQ(arr[3], std::int32_t(2));
            }
            SUBCASE("with mapping")
            {
                std::vector<std::size_t> child_index_to_type_id{1, 0};
                // type ids
                dense_union_array::type_id_buffer_type type_ids{
                    {std::uint8_t(1), std::uint8_t(0), std::uint8_t(1), std::uint8_t(0)}
                };

                // the array
                dense_union_array arr(
                    std::move(children),
                    std::move(type_ids),
                    std::move(offsets),
                    std::make_optional(std::move(child_index_to_type_id))
                );

                // check the size
                REQUIRE_EQ(arr.size(), 4);

                // check elements have values
                CHECK(arr[0].has_value());
                CHECK(!arr[1].has_value());
                CHECK(arr[2].has_value());
                CHECK(arr[3].has_value());

                CHECK_NULLABLE_VARIANT_EQ(arr[0], std::int16_t(1));
                CHECK_NULLABLE_VARIANT_EQ(arr[2], std::int16_t(0));
                CHECK_NULLABLE_VARIANT_EQ(arr[3], std::int32_t(2));
            }
        }
        TEST_CASE("basics")
        {
            const std::string format_string = "+ud:3,4";
            const std::size_t n_c = 2;
            const std::size_t n = 4;

            auto proxy = test::make_dense_union_proxy(format_string, n_c);
            dense_union_array uarr(std::move(proxy));

            REQUIRE(uarr.size() == n);

            SUBCASE("copy")
            {
#ifdef SPARROW_TRACK_COPIES
                copy_tracker::reset(copy_tracker::key<dense_union_array>());
#endif
                dense_union_array uarr2(uarr);
                CHECK_EQ(uarr2, uarr);
#ifdef SPARROW_TRACK_COPIES
                CHECK_EQ(copy_tracker::count(copy_tracker::key<dense_union_array>()), 1);
#endif

                dense_union_array uarr3(test::make_dense_union_proxy(format_string, n_c, true));
                CHECK_NE(uarr3, uarr);
                uarr3 = uarr;
                CHECK_EQ(uarr3, uarr);
            }

            SUBCASE("move")
            {
                dense_union_array uarr2(uarr);
                dense_union_array uarr3(std::move(uarr2));
                CHECK_EQ(uarr3, uarr);

                dense_union_array uarr4(test::make_dense_union_proxy(format_string, n_c, true));
                CHECK_NE(uarr4, uarr);
                uarr4 = std::move(uarr3);
                CHECK_EQ(uarr4, uarr);
            }

            SUBCASE("operator[]")
            {
                for (std::size_t i = 0; i < n; ++i)
                {
                    const auto& val = uarr[i];
                    REQUIRE(val.has_value());
                }
            }

#if SPARROW_GCC_11_2_WORKAROUND
            using variant_type = std::decay_t<decltype(uarr[0])>;
            using base_type = typename variant_type::base_type;
#endif

            // 0
            std::visit(
                [](auto&& arg)
                {
                    using inner_type = std::decay_t<typename std::decay_t<decltype(arg)>::value_type>;
                    if constexpr (std::is_same_v<inner_type, float32_t>)
                    {
                        REQUIRE_EQ(0.0f, arg.value());
                    }
                    else
                    {
                        CHECK(false);
                    }
                },
#if SPARROW_GCC_11_2_WORKAROUND
                static_cast<const base_type&>(uarr[0])
#else
                uarr[0]
#endif
            );

            // 1
            std::visit(
                [](auto&& arg)
                {
                    using inner_type = std::decay_t<typename std::decay_t<decltype(arg)>::value_type>;
                    if constexpr (std::is_same_v<inner_type, std::uint16_t>)
                    {
                        REQUIRE_EQ(0, arg.value());
                    }
                    else
                    {
                        CHECK(false);
                    }
                },
#if SPARROW_GCC_11_2_WORKAROUND
                static_cast<const base_type&>(uarr[1])
#else
                uarr[1]
#endif
            );

            // 2
            std::visit(
                [](auto&& arg)
                {
                    using inner_type = std::decay_t<typename std::decay_t<decltype(arg)>::value_type>;
                    if constexpr (std::is_same_v<inner_type, float32_t>)
                    {
                        REQUIRE_EQ(1.0f, arg.value());
                    }
                    else
                    {
                        CHECK(false);
                    }
                },
#if SPARROW_GCC_11_2_WORKAROUND
                static_cast<const base_type&>(uarr[2])
#else
                uarr[2]
#endif
            );

            // 3
            std::visit(
                [](auto&& arg)
                {
                    using inner_type = std::decay_t<typename std::decay_t<decltype(arg)>::value_type>;
                    if constexpr (std::is_same_v<inner_type, std::uint16_t>)
                    {
                        REQUIRE_EQ(1, arg.value());
                    }
                    else
                    {
                        CHECK(false);
                    }
                },
#if SPARROW_GCC_11_2_WORKAROUND
                static_cast<const base_type&>(uarr[3])
#else
                uarr[3]
#endif
            );
        }

        TEST_CASE("mutation")
        {
            auto proxy = test::make_dense_union_proxy("+ud:3,4", 2);
            dense_union_array uarr(std::move(proxy));

            uarr.insert(
                uarr.cbegin() + 1,
                array_traits::value_type{make_nullable(float32_t(9.0f))}
            );

            REQUIRE_EQ(uarr.size(), 5);
            CHECK_NULLABLE_VARIANT_EQ(uarr[1], float32_t(9.0f));
            const auto& proxy_after_insert = detail::array_access::get_arrow_proxy(uarr);
            REQUIRE_EQ(proxy_after_insert.children().size(), 2);
            CHECK_EQ(proxy_after_insert.children()[0].length(), 3);
            CHECK_EQ(proxy_after_insert.children()[1].length(), 2);

            const auto& offsets = proxy_after_insert.buffers()[1];
            const auto* offset_data = reinterpret_cast<const std::int32_t*>(offsets.data());
            CHECK_EQ(offset_data[0], 0);
            CHECK_EQ(offset_data[1], 1);
            CHECK_EQ(offset_data[2], 0);
            CHECK_EQ(offset_data[3], 2);
            CHECK_EQ(offset_data[4], 1);
            CHECK_EQ(proxy_after_insert.format(), "+ud:3,4");

            uarr.erase(uarr.cbegin() + 1);
            CHECK_EQ(uarr.size(), 4);
            CHECK_NULLABLE_VARIANT_EQ(uarr[1], std::uint16_t(0));
            CHECK_EQ(detail::array_access::get_arrow_proxy(uarr).children()[0].length(), 2);

            array erased_union(std::move(uarr));
            array source_union(erased_union);
            erased_union.insert(
                erased_union.cbegin() + 2,
                source_union.cbegin(),
                source_union.cbegin() + 1
            );
            CHECK_EQ(erased_union.size(), 5);
            CHECK_NULLABLE_VARIANT_EQ(erased_union[2], float32_t(0.0f));
        }

        TEST_CASE("mutation edge cases")
        {
            SUBCASE("zero-count and empty-range insertion are no-ops")
            {
                dense_union_array uarr(test::make_dense_union_proxy("+ud:3,4", 2));
                std::vector<array_traits::value_type> empty_values;

                uarr.insert(uarr.cbegin() + 1, uarr[0], 0);
                uarr.insert(uarr.cbegin() + 1, empty_values.cbegin(), empty_values.cend());

                CHECK_EQ(uarr.size(), 4);
                CHECK_EQ(detail::array_access::get_arrow_proxy(uarr).format(), "+ud:3,4");
            }

            SUBCASE("caches repeated new alternatives")
            {
                dense_union_array uarr(test::make_dense_union_proxy("+ud:3,4", 2));
                std::vector<array_traits::value_type> values{
                    make_nullable(std::int32_t(7)),
                    make_nullable(std::int32_t(8))
                };

                uarr.insert(uarr.cend(), values.cbegin(), values.cend());

                CHECK_EQ(detail::array_access::get_arrow_proxy(uarr).format(), "+ud:3,4,0");
                CHECK_EQ(detail::array_access::get_arrow_proxy(uarr).children().size(), 3);
                CHECK_EQ(detail::array_access::get_arrow_proxy(uarr).children()[2].length(), 2);
                CHECK_NULLABLE_VARIANT_EQ(uarr[4], std::int32_t(7));
                CHECK_NULLABLE_VARIANT_EQ(uarr[5], std::int32_t(8));
            }

            SUBCASE("supports borrowed values and self-insertion")
            {
                dense_union_array uarr(test::make_dense_union_proxy("+ud:3,4", 2));
                uarr.insert(uarr.cend(), uarr[0]);

                array dynamic(std::move(uarr));
                dynamic.insert(dynamic.cbegin() + 1, dynamic.cbegin(), dynamic.cbegin() + 1);

                CHECK_EQ(dynamic.size(), 6);
                CHECK_NULLABLE_VARIANT_EQ(dynamic[1], float32_t(0.0f));
            }
        }

        TEST_CASE("rebuilds nested and noncanonical children")
        {
            auto list_child = list_array(
                array(primitive_array<std::int16_t>{std::vector<std::int16_t>{1, 2, 3}}),
                std::vector<std::int32_t>{0, 1, 3},
                true
            );
            auto large_string_child = big_string_array(std::vector<std::string>{"alpha", "beta"}, true);

            dense_union_array uarr(
                std::vector<array>{array(std::move(list_child)), array(std::move(large_string_child))},
                dense_union_array::type_id_buffer_type{std::vector<std::uint8_t>{0, 1, 0, 1}},
                dense_union_array::offset_buffer_type{std::vector<std::uint32_t>{0, 0, 1, 1}}
            );

            uarr.erase(uarr.cbegin() + 1);

            const auto& proxy = detail::array_access::get_arrow_proxy(uarr);
            CHECK_EQ(proxy.children()[0].format(), "+l");
            CHECK_EQ(proxy.children()[1].format(), "U");
            CHECK_EQ(proxy.children()[0].length(), 2);
            CHECK_EQ(proxy.children()[1].length(), 1);
            CHECK(uarr[0].has_value());
            CHECK(uarr[1].has_value());
            CHECK(uarr[2].has_value());
            array large_string_result{proxy.children()[1].view()};
            CHECK_NULLABLE_VARIANT_EQ(large_string_result[0], std::string_view("beta"));
        }

        TEST_CASE("preserves a nested union child during rebuild")
        {
            auto nested_union = dense_union_array(
                std::vector<array>{
                    array(primitive_array<std::int32_t>{std::vector<std::int32_t>{10}}),
                    array(string_array{std::vector<std::string>{"inner"}})
                },
                dense_union_array::type_id_buffer_type{std::vector<std::uint8_t>{0, 1}},
                dense_union_array::offset_buffer_type{std::vector<std::uint32_t>{0, 0}}
            );
            auto large_string_child = big_string_array(std::vector<std::string>{"alpha", "beta"}, true);

            dense_union_array uarr(
                std::vector<array>{array(std::move(nested_union)), array(std::move(large_string_child))},
                dense_union_array::type_id_buffer_type{std::vector<std::uint8_t>{0, 1, 0, 1}},
                dense_union_array::offset_buffer_type{std::vector<std::uint32_t>{0, 0, 1, 1}}
            );

            uarr.erase(uarr.cbegin() + 1);

            const auto& proxy = detail::array_access::get_arrow_proxy(uarr);
            CHECK_EQ(proxy.children()[0].format(), "+ud:0,1");
            CHECK_EQ(proxy.children()[0].length(), 2);
            CHECK_EQ(proxy.children()[1].format(), "U");
            CHECK_EQ(proxy.children()[1].length(), 1);
            CHECK(uarr[0].has_value());
            CHECK(uarr[1].has_value());
            CHECK(uarr[2].has_value());
            array nested_union_result{proxy.children()[0].view()};
            CHECK_NULLABLE_VARIANT_EQ(nested_union_result[0], std::int32_t(10));
            CHECK_NULLABLE_VARIANT_EQ(nested_union_result[1], std::string_view("inner"));
        }

        TEST_CASE("distinguishes parameterized child schemas")
        {
            dense_union_array decimal_union(
                std::vector<array>{
                    array(decimal_64_array{std::vector<std::int64_t>{100}, std::size_t{18}, 2})
                },
                dense_union_array::type_id_buffer_type{std::vector<std::uint8_t>{0}},
                dense_union_array::offset_buffer_type{std::vector<std::uint32_t>{0}}
            );
            std::vector<array_traits::value_type> decimal_values{
                make_nullable(decimal<std::int64_t>{123, 3}),
                make_nullable(decimal<std::int64_t>{456, 4})
            };

            decimal_union.insert(decimal_union.cend(), decimal_values.cbegin(), decimal_values.cend());

            const auto& decimal_proxy = detail::array_access::get_arrow_proxy(decimal_union);
            REQUIRE_EQ(decimal_proxy.children().size(), 3);
            CHECK_EQ(decimal_proxy.children()[0].format(), "d:18,2,64");
            CHECK_EQ(decimal_proxy.children()[1].format(), "d:18,3,64");
            CHECK_EQ(decimal_proxy.children()[2].format(), "d:18,4,64");

            const auto* new_york = date::locate_zone("America/New_York");
            const auto* utc = date::locate_zone("UTC");
            const auto* los_angeles = date::locate_zone("America/Los_Angeles");
            dense_union_array timestamp_union(
                std::vector<array>{
                    array(timestamp_seconds_array{
                        new_york,
                        std::vector<timestamp_second>{
                            timestamp_second{new_york, date::sys_seconds{std::chrono::seconds{0}}}
                        }
                    })
                },
                dense_union_array::type_id_buffer_type{std::vector<std::uint8_t>{0}},
                dense_union_array::offset_buffer_type{std::vector<std::uint32_t>{0}}
            );
            std::vector<array_traits::value_type> timestamp_values{
                make_nullable(timestamp_second{utc, date::sys_seconds{std::chrono::seconds{1}}}),
                make_nullable(timestamp_second{los_angeles, date::sys_seconds{std::chrono::seconds{2}}})
            };

            timestamp_union.insert(timestamp_union.cend(), timestamp_values.cbegin(), timestamp_values.cend());

            const auto& timestamp_proxy = detail::array_access::get_arrow_proxy(timestamp_union);
            REQUIRE_EQ(timestamp_proxy.children().size(), 3);
            CHECK(timestamp_proxy.children()[0].format() != timestamp_proxy.children()[1].format());
            CHECK(timestamp_proxy.children()[1].format() != timestamp_proxy.children()[2].format());
            CHECK(timestamp_proxy.children()[1].format().find("UTC") != std::string_view::npos);
            CHECK(timestamp_proxy.children()[2].format().find("America/Los_Angeles") != std::string_view::npos);
        }

#if defined(__cpp_lib_format)
        TEST_CASE("formatting")
        {
            const std::string format_string = "+ud:3,4";
            const std::size_t n_c = 2;

            auto proxy = test::make_dense_union_proxy(format_string, n_c);
            dense_union_array uarr(std::move(proxy));

            const std::string formatted = std::format("{}", uarr);
            constexpr std::string_view expected = "DenseUnion [name=test | size=4] <0, 0, 1, 1>";
            CHECK_EQ(formatted, expected);
        }
#endif
    }
}
