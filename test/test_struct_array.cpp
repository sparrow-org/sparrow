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
#include <sstream>
#include <stdexcept>
#include <string_view>

#include "sparrow/array.hpp"
#include "sparrow/layout/array_helper.hpp"
#include "sparrow/map_array.hpp"
#include "sparrow/primitive_array.hpp"
#include "sparrow/struct_array.hpp"
#include "sparrow/variable_size_binary_array.hpp"

#include "../test/external_array_data_creation.hpp"
#include "doctest/doctest.h"
#include "test_utils.hpp"

namespace sparrow
{
    namespace test
    {
        template <class T0, class T1>
        arrow_proxy make_struct_proxy(size_t n)
        {
            // n-children == 2
            std::vector<ArrowArray> children_arrays(2);
            std::vector<ArrowSchema> children_schemas(2);

            test::fill_schema_and_array<T0>(children_schemas[0], children_arrays[0], n, 0, {});
            children_schemas[0].name = "item 0";

            test::fill_schema_and_array<T1>(children_schemas[1], children_arrays[1], n, 0, {});
            children_schemas[1].name = "item 1";

            ArrowArray arr{};
            ArrowSchema schema{};
            test::fill_schema_and_array_for_struct_layout(
                schema,
                arr,
                std::move(children_schemas),
                std::move(children_arrays),
                {}
            );
            return arrow_proxy(std::move(arr), std::move(schema));
        }
    }

    TEST_SUITE("struct_array")
    {
        static_assert(is_struc_array_v<struct_array>);

        TEST_CASE("constructors")
        {
            primitive_array<std::int16_t> flat_arr(
                {{std::int16_t(0), std::int16_t(1), std::int16_t(2), std::int16_t(3)}, true, "flat_arr1"}
            );
            primitive_array<float32_t> flat_arr2({{4.0f, 5.0f, 6.0f, 7.0f}, true, "flat_arr2"});
            primitive_array<std::int32_t> flat_arr3(
                {{std::int32_t(8), std::int32_t(9), std::int32_t(10), std::int32_t(11)}, true, "flat_arr3"}
            );

            // detyped arrays
            array arr1(std::move(flat_arr));
            array arr2(std::move(flat_arr2));
            array arr3(std::move(flat_arr3));
            std::vector<array> children{std::move(arr1), std::move(arr2), std::move(arr3)};

            SUBCASE("with whildren, nullable, name and metadata")
            {
                const struct_array arr(children, false, "name", metadata_sample_opt);

                // check the size
                REQUIRE_EQ(arr.size(), 4);

                // check the children
                REQUIRE_EQ(arr[0].value().size(), 3);
                REQUIRE_EQ(arr[1].value().size(), 3);
                REQUIRE_EQ(arr[2].value().size(), 3);
                REQUIRE_EQ(arr[3].value().size(), 3);

                // check the values
                const auto child0 = arr[0].value();
                const auto child0_names = child0.names();
                REQUIRE_EQ(child0_names.size(), 3);
                CHECK_EQ(child0_names[0], "flat_arr1");
                CHECK_EQ(child0_names[1], "flat_arr2");
                CHECK_EQ(child0_names[2], "flat_arr3");

                CHECK_NULLABLE_VARIANT_EQ(child0[0], std::int16_t(0));
                CHECK_NULLABLE_VARIANT_EQ(child0[1], float32_t(4.0f));
                CHECK_NULLABLE_VARIANT_EQ(child0[2], std::int32_t(8));

                const auto child1 = arr[1].value();
                const auto child1_names = child1.names();
                REQUIRE_EQ(child1_names.size(), 3);
                CHECK_EQ(child1_names[0], "flat_arr1");
                CHECK_EQ(child1_names[1], "flat_arr2");
                CHECK_EQ(child1_names[2], "flat_arr3");
                CHECK_NULLABLE_VARIANT_EQ(child1[0], std::int16_t(1));
                CHECK_NULLABLE_VARIANT_EQ(child1[1], float32_t(5.0f));
                CHECK_NULLABLE_VARIANT_EQ(child1[2], std::int32_t(9));

                const auto child2 = arr[2].value();
                const auto child2_names = child2.names();
                REQUIRE_EQ(child2_names.size(), 3);
                CHECK_EQ(child2_names[0], "flat_arr1");
                CHECK_EQ(child2_names[1], "flat_arr2");
                CHECK_EQ(child2_names[2], "flat_arr3");
                CHECK_NULLABLE_VARIANT_EQ(child2[0], std::int16_t(2));
                CHECK_NULLABLE_VARIANT_EQ(child2[1], float32_t(6.0f));
                CHECK_NULLABLE_VARIANT_EQ(child2[2], std::int32_t(10));

                const auto child3 = arr[3].value();
                const auto child3_names = child3.names();
                REQUIRE_EQ(child3_names.size(), 3);
                CHECK_EQ(child3_names[0], "flat_arr1");
                CHECK_EQ(child3_names[1], "flat_arr2");
                CHECK_EQ(child3_names[2], "flat_arr3");
                CHECK_NULLABLE_VARIANT_EQ(child3[0], std::int16_t(3));
                CHECK_NULLABLE_VARIANT_EQ(child3[1], float32_t(7.0f));
                CHECK_NULLABLE_VARIANT_EQ(child3[2], std::int32_t(11));
            }

            SUBCASE("with whildren, bitmap, name and metadata")
            {
                std::vector<bool> bitmap{true, false, true, false};
                const struct_array arr(children, bitmap, "name", metadata_sample_opt);

                // check the size
                REQUIRE_EQ(arr.size(), 4);

                // check the children
                REQUIRE_EQ(arr[0].value().size(), 3);
                REQUIRE_EQ(arr[2].value().size(), 3);

                // check the values
                const auto child0 = arr[0].value();
                CHECK_NULLABLE_VARIANT_EQ(child0[0], std::int16_t(0));
                CHECK_NULLABLE_VARIANT_EQ(child0[1], float32_t(4.0f));
                CHECK_NULLABLE_VARIANT_EQ(child0[2], std::int32_t(8));

                CHECK_FALSE(arr[1].has_value());

                const auto child2 = arr[2].value();

                CHECK_NULLABLE_VARIANT_EQ(child2[0], std::int16_t(2));
                CHECK_NULLABLE_VARIANT_EQ(child2[1], float32_t(6.0f));
                CHECK_NULLABLE_VARIANT_EQ(child2[2], std::int32_t(10));

                CHECK_FALSE(arr[3].has_value());
            }
        }

        TEST_CASE("mutation")
        {
            struct_array arr(test::make_struct_proxy<std::int32_t, std::uint8_t>(3));
            struct_array source(test::make_struct_proxy<std::int32_t, std::uint8_t>(3));
            const auto inserted_struct = source[1].value();

            arr.insert(arr.cbegin() + 1, make_nullable(inserted_struct));
            REQUIRE_EQ(arr.size(), 4);
            CHECK_EQ(arr.raw_child(0)->get_arrow_proxy().length(), 4);
            CHECK_EQ(arr.raw_child(1)->get_arrow_proxy().length(), 4);
            CHECK_NULLABLE_VARIANT_EQ(arr[1].value()[0], std::int32_t(1));

            arr.insert(arr.cbegin() + 2, make_nullable(struct_value{}, false));
            REQUIRE_EQ(arr.size(), 5);
            CHECK_FALSE(arr[2].has_value());
            CHECK_EQ(arr.raw_child(0)->get_arrow_proxy().length(), 5);
            CHECK_EQ(arr.raw_child(1)->get_arrow_proxy().length(), 5);

            arr.erase(arr.cbegin() + 1, arr.cbegin() + 3);
            CHECK_EQ(arr.size(), 3);
            CHECK_EQ(arr.raw_child(0)->get_arrow_proxy().length(), 3);
            CHECK_EQ(arr.raw_child(1)->get_arrow_proxy().length(), 3);

            arr.resize(5, make_nullable(source[0].value()));
            CHECK_EQ(arr.size(), 5);
            CHECK_NULLABLE_VARIANT_EQ(arr[4].value()[0], std::int32_t(0));
            arr.resize(2, make_nullable(struct_value{}, false));
            CHECK_EQ(arr.size(), 2);

            array erased_target(test::make_struct_proxy<std::int32_t, std::uint8_t>(3));
            array erased_source(test::make_struct_proxy<std::int32_t, std::uint8_t>(3));
            erased_target.insert(
                erased_target.cend(),
                erased_source.cbegin(),
                erased_source.cbegin() + 1
            );
            CHECK_EQ(erased_target.size(), 4);
        }

        TEST_CASE("mutation edge cases")
        {
            SUBCASE("zero-count insertion and empty erasure are no-ops")
            {
                struct_array arr(test::make_struct_proxy<std::int32_t, std::uint8_t>(3));
                const auto original_size = arr.size();
                std::vector<nullable<struct_value>> empty_values;

                arr.insert(arr.cbegin() + 1, make_nullable(struct_value{}), 0);
                arr.insert(arr.cbegin() + 1, empty_values.cbegin(), empty_values.cend());
                // erase of an empty range returns the position without touching the array.
                const std::size_t size_before_empty_erase = arr.size();
                arr.erase(arr.cbegin() + 1, arr.cbegin() + 1);
                CHECK_EQ(arr.size(), size_before_empty_erase);

                CHECK_EQ(arr.size(), original_size);
            }

            SUBCASE("rejects mutations of sliced arrays")
            {
                struct_array source(test::make_struct_proxy<std::int32_t, std::uint8_t>(3));
                struct_array sliced(detail::array_access::get_arrow_proxy(source).slice(1, 3));

                CHECK_THROWS_AS(
                    sliced.insert(sliced.cbegin(), make_nullable(struct_value{})),
                    std::logic_error
                );
                CHECK_THROWS_AS(sliced.erase(sliced.cbegin()), std::logic_error);
                CHECK_THROWS_AS(sliced.set_value(0, source[0].value()), std::logic_error);
                CHECK_THROWS_AS(sliced[0].value() = source[0].value(), std::logic_error);
            }

            SUBCASE("rejects an incompatible field count")
            {
                struct_array arr(test::make_struct_proxy<std::int32_t, std::uint8_t>(3));
                struct_array incompatible(
                    std::vector<array>{array(primitive_array<std::int32_t>{std::vector<std::int32_t>{1}})}
                );

                CHECK_THROWS_AS(
                    arr.insert(arr.cend(), make_nullable(incompatible[0].value())),
                    std::invalid_argument
                );
            }

            SUBCASE("a failing field leaves the record unchanged")
            {
                // Two fields whose types differ, plus a donor whose second field is a string:
                // the mismatch is only detected on the second field, so a naive loop would
                // already have written the first. The pre-flight check must prevent that.
                std::vector<nullable<std::int32_t>> first{std::int32_t(11)};
                std::vector<nullable<std::int64_t>> second{std::int64_t(22)};
                struct_array arr(
                    std::vector<array>{
                        array(primitive_array<std::int32_t>(std::move(first))),
                        array(primitive_array<std::int64_t>(std::move(second))),
                    }
                );

                std::vector<nullable<std::int32_t>> donor_first{std::int32_t(99)};
                std::vector<nullable<std::string>> donor_second{std::string("mismatch")};
                struct_array donor(
                    std::vector<array>{
                        array(primitive_array<std::int32_t>(std::move(donor_first))),
                        array(string_array(std::move(donor_second))),
                    }
                );

                CHECK_THROWS_AS(arr.set_value(0, donor[0].value()), std::invalid_argument);
                // Neither field may have been written.
                CHECK_NULLABLE_VARIANT_EQ(arr[0].value()[0], std::int32_t(11));
                CHECK_NULLABLE_VARIANT_EQ(arr[0].value()[1], std::int64_t(22));
            }

            SUBCASE("a default-constructed struct_value assigns field defaults, not null")
            {
                struct_array arr(test::make_struct_proxy<std::int32_t, std::uint8_t>(2));
                arr[0] = struct_value{};
                // The element stays valid: the fields take their type defaults.
                REQUIRE(arr[0].has_value());
                CHECK_NULLABLE_VARIANT_EQ(arr[0].value()[0], std::int32_t(0));
                CHECK_NULLABLE_VARIANT_EQ(arr[0].value()[1], std::uint8_t(0));
            }

            SUBCASE("a map child does not claim in-place assignment")
            {
                // map_array::operator[] yields nullable<map_value> by value and map_value is a
                // non-owning view, so writing through it would be silently lost. It must not be
                // reported as writable in place.
                map_array map_child(
                    array(primitive_array<std::int32_t>(std::vector<nullable<std::int32_t>>{std::int32_t(1)})),
                    array(primitive_array<std::int32_t>(std::vector<nullable<std::int32_t>>{std::int32_t(2)})),
                    map_array::offset_from_sizes(std::vector<std::size_t>{1})
                );
                CHECK_FALSE(supports_element_assignment(array(&map_child)));
            }

            SUBCASE("in-place element assignment")
            {
                struct_array arr(test::make_struct_proxy<std::int32_t, std::uint8_t>(3));

                // Assigning a reference from the same array.
                arr[0] = arr[1];
                REQUIRE(arr[0].has_value());
                CHECK_NULLABLE_VARIANT_EQ(arr[0].value()[0], std::int32_t(1));
                CHECK_NULLABLE_VARIANT_EQ(arr[0].value()[1], std::uint8_t(1));
                CHECK_EQ(arr.size(), 3);
                CHECK_EQ(arr.raw_child(0)->get_arrow_proxy().length(), 3);
                CHECK_EQ(arr.raw_child(1)->get_arrow_proxy().length(), 3);

                // Assigning a value from another array.
                struct_array source(test::make_struct_proxy<std::int32_t, std::uint8_t>(3));
                arr[2] = source[2];
                CHECK_NULLABLE_VARIANT_EQ(arr[2].value()[0], std::int32_t(2));
                CHECK_NULLABLE_VARIANT_EQ(arr[2].value()[1], std::uint8_t(2));

                // Assigning null through the proxy and assigning a value again.
                arr[1] = nullval;
                CHECK_FALSE(arr[1].has_value());
                arr[1] = arr[0];
                CHECK(arr[1].has_value());
                CHECK_NULLABLE_VARIANT_EQ(arr[1].value()[0], std::int32_t(1));

                // Assigning a null struct value keeps the element null.
                arr[0] = make_nullable(struct_value{}, false);
                CHECK_FALSE(arr[0].has_value());
            }

            SUBCASE("assignment writes the children in place")
            {
                struct_array arr(test::make_struct_proxy<std::int32_t, std::uint8_t>(4));

                // The value of one element changes: the children must keep their buffers rather
                // than be rebuilt, which is what an O(size) implementation does. Comparing the
                // child data pointers pins that: a rebuild installs fresh buffers.
                const std::int32_t*
                    child_0_before = arr.raw_child(0)->get_arrow_proxy().buffers()[1].data<std::int32_t>();
                const std::uint8_t*
                    child_1_before = arr.raw_child(1)->get_arrow_proxy().buffers()[1].data<std::uint8_t>();

                arr[2] = arr[0];

                CHECK_EQ(arr.raw_child(0)->get_arrow_proxy().buffers()[1].data<std::int32_t>(), child_0_before);
                CHECK_EQ(arr.raw_child(1)->get_arrow_proxy().buffers()[1].data<std::uint8_t>(), child_1_before);
                CHECK_NULLABLE_VARIANT_EQ(arr[2].value()[0], std::int32_t(0));
                CHECK_NULLABLE_VARIANT_EQ(arr[2].value()[1], std::uint8_t(0));

                // Only the assigned element changed in each child.
                CHECK_NULLABLE_VARIANT_EQ(arr[0].value()[0], std::int32_t(0));
                CHECK_NULLABLE_VARIANT_EQ(arr[1].value()[0], std::int32_t(1));
                CHECK_NULLABLE_VARIANT_EQ(arr[3].value()[0], std::int32_t(3));
            }

            SUBCASE("assignment refreshes a child whose buffers are replaced")
            {
                // A string child replaces its data buffer when the assigned string does not fit:
                // the assignment must leave the other elements readable and unchanged.
                std::vector<array> children;
                children.emplace_back(string_array{std::vector<nullable<std::string>>{"a", "bb", "ccc"}});
                children.emplace_back(primitive_array<std::int64_t>{std::vector<std::int64_t>{1, 2, 3}});
                struct_array arr{children};

                // Three elements: the child reallocates while the array holds several elements,
                // which is where the stale per-element views become visible.
                const std::string long_value(200, 'y');
                std::vector<array> wide_children;
                wide_children.emplace_back(
                    string_array{std::vector<nullable<std::string>>{long_value, long_value, long_value}}
                );
                wide_children.emplace_back(primitive_array<std::int64_t>{std::vector<std::int64_t>{7, 8, 9}});
                struct_array wide{wide_children};

                // Read every element before the assignment, as a user iterating the array would.
                for (std::size_t r = 0; r < arr.size(); ++r)
                {
                    (void) arr[r].value()[0];
                    (void) arr[r].value()[1];
                }

                arr[1] = wide[1];

                const auto assigned = std::get<nullable<std::string_view>>(arr[1].value()[0]).value();
                CHECK_EQ(assigned.size(), long_value.size());

                // The elements that were not assigned must still read their original values.
                CHECK_EQ(std::get<nullable<std::string_view>>(arr[0].value()[0]).value(), std::string_view("a"));
                CHECK_EQ(std::get<nullable<std::string_view>>(arr[2].value()[0]).value(), std::string_view("ccc"));
                CHECK_NULLABLE_VARIANT_EQ(arr[0].value()[1], std::int64_t(1));
                CHECK_NULLABLE_VARIANT_EQ(arr[2].value()[1], std::int64_t(3));
            }

            SUBCASE("assignment works for a nested struct child")
            {
                // A nested struct child cannot be rebuilt from its elements (array_make_from_element
                // rejects nested values), so this only works through the in-place path.
                const auto make = []
                {
                    std::vector<array> grand_children;
                    grand_children.push_back(
                        array(primitive_array<std::int64_t>{std::vector<std::int64_t>{10, 20, 30}})
                    );
                    std::vector<array> children;
                    children.push_back(array(struct_array{grand_children}));
                    return struct_array{children};
                };

                struct_array arr = make();
                for (std::size_t r = 0; r < arr.size(); ++r)
                {
                    (void) arr[r].value()[0];
                }

                arr[1] = arr[2];

                // The child of the nested struct holds the primitive value.
                const auto nested = std::get<nullable<struct_value>>(arr[1].value()[0]);
                REQUIRE(nested.has_value());
                CHECK_NULLABLE_VARIANT_EQ(nested.value()[0], std::int64_t(30));
                for (std::size_t r = 0; r < arr.size(); ++r)
                {
                    (void) arr[r].value()[0];
                }
            }

            SUBCASE("a child that cannot be written in place takes the rebuild fallback")
            {
                // decimal_array is deliberately excluded from the in-place path (its element
                // assignment does not compile with MSVC for int128_t / int256_t), so a struct
                // holding one must still assign correctly through rebuild_children.
                const auto make = []
                {
                    std::vector<array> children;
                    children.push_back(array(primitive_array<std::int64_t>{std::vector<std::int64_t>{1, 2, 3}}));
                    children.push_back(array(
                        decimal_array<decimal<std::int32_t>>{std::vector<std::int32_t>{10, 20, 30}, std::size_t(2), 0}
                    ));
                    return struct_array{children};
                };

                struct_array arr = make();

                // The decimal child cannot be written in place, so this assignment goes through
                // the rebuild fallback: the result must still be correct.
                arr[0] = arr[2];

                CHECK_NULLABLE_VARIANT_EQ(arr[0].value()[0], std::int64_t(3));
                CHECK_NULLABLE_VARIANT_EQ(arr[1].value()[0], std::int64_t(2));
                CHECK_EQ(arr.size(), 3);
            }

            SUBCASE("non-const iteration yields mutable proxies")
            {
                struct_array arr(test::make_struct_proxy<std::int32_t, std::uint8_t>(3));
                std::int32_t expected = 0;
                for (auto element : arr)
                {
                    REQUIRE(element.has_value());
                    CHECK_NULLABLE_VARIANT_EQ(element.value()[0], expected);
                    ++expected;
                }
                CHECK_EQ(expected, 3);

                auto it = arr.begin();
                *it = arr[1];
                CHECK_NULLABLE_VARIANT_EQ(arr[0].value()[0], std::int32_t(1));
                CHECK_NULLABLE_VARIANT_EQ(arr[0].value()[1], std::uint8_t(1));
            }

            SUBCASE("in-place assignment rejects an incompatible field count")
            {
                struct_array arr(test::make_struct_proxy<std::int32_t, std::uint8_t>(3));
                struct_array incompatible(
                    std::vector<array>{array(primitive_array<std::int32_t>{std::vector<std::int32_t>{1}})}
                );
                CHECK_THROWS_AS(arr.set_value(0, incompatible[0].value()), std::invalid_argument);
                CHECK_THROWS_AS(arr[0] = incompatible[0], std::invalid_argument);
            }
        }
    };

    TEST_CASE("struct_reference")
    {
        // Two children named "item 0" and "item 1", values 0..n-1 in each (see make_struct_proxy).
        const auto make = []
        {
            return struct_array(test::make_struct_proxy<std::int32_t, std::uint8_t>(3));
        };

        SUBCASE("size and empty")
        {
            struct_array arr = make();
            auto reference = arr[0].value();
            CHECK_EQ(reference.size(), 2);
            CHECK_FALSE(reference.empty());

            // A struct with zero children is the empty record: size() is 0 and empty() is true.
            // struct_array itself requires at least one child, so this is checked on the value.
            const struct_value empty_value{};
            CHECK_EQ(empty_value.size(), 0);
            CHECK(empty_value.empty());
        }

        SUBCASE("element access")
        {
            struct_array arr = make();
            auto reference = arr[1].value();

            CHECK_NULLABLE_VARIANT_EQ(reference[0], std::int32_t(1));
            CHECK_NULLABLE_VARIANT_EQ(reference[1], std::uint8_t(1));

            CHECK_NULLABLE_VARIANT_EQ(reference.at(0), std::int32_t(1));
            CHECK_THROWS_AS(reference.at(2), std::out_of_range);

            CHECK_NULLABLE_VARIANT_EQ(reference.at("item 0"), std::int32_t(1));
            CHECK_THROWS_AS(reference.at("missing"), std::out_of_range);

            CHECK_NULLABLE_VARIANT_EQ(reference.front(), std::int32_t(1));
            CHECK_NULLABLE_VARIANT_EQ(reference.back(), std::uint8_t(1));
        }

        SUBCASE("forward iteration")
        {
            struct_array arr = make();
            auto reference = arr[2].value();

            std::vector<array_traits::value_type> forward(reference.begin(), reference.end());
            REQUIRE_EQ(forward.size(), 2);
            CHECK_NULLABLE_VARIANT_EQ(forward[0], std::int32_t(2));
            CHECK_NULLABLE_VARIANT_EQ(forward[1], std::uint8_t(2));

            std::vector<array_traits::value_type> const_forward(reference.cbegin(), reference.cend());
            CHECK_EQ(forward, const_forward);
        }

        SUBCASE("reverse iteration")
        {
            struct_array arr = make();
            auto reference = arr[2].value();

            std::vector<array_traits::value_type> reverse(reference.rbegin(), reference.rend());
            REQUIRE_EQ(reverse.size(), 2);
            // Reversed: the last child comes first.
            CHECK_NULLABLE_VARIANT_EQ(reverse[0], std::uint8_t(2));
            CHECK_NULLABLE_VARIANT_EQ(reverse[1], std::int32_t(2));

            std::vector<array_traits::value_type> const_reverse(reference.crbegin(), reference.crend());
            CHECK_EQ(reverse, const_reverse);
        }

        SUBCASE("names")
        {
            struct_array arr = make();
            auto reference = arr[0].value();

            // names() yields the child names as optional string views.
            std::vector<std::optional<std::string_view>> collected;
            for (const auto& name : reference.names())
            {
                collected.push_back(name);
            }
            REQUIRE_EQ(collected.size(), 2);
            REQUIRE(collected[0].has_value());
            REQUIRE(collected[1].has_value());
            CHECK_EQ(*collected[0], std::string_view("item 0"));
            CHECK_EQ(*collected[1], std::string_view("item 1"));
        }

        SUBCASE("conversion to struct_value")
        {
            struct_array arr = make();
            auto reference = arr[1].value();

            const struct_value value = static_cast<struct_value>(reference);
            CHECK_EQ(value.size(), 2);
            CHECK_NULLABLE_VARIANT_EQ(value[0], std::int32_t(1));
            CHECK_NULLABLE_VARIANT_EQ(value[1], std::uint8_t(1));
            // The reference reads the same record as the value it converts to.
            CHECK(reference == value);
        }

        SUBCASE("equality")
        {
            struct_array arr = make();

            // Same element: equal.
            CHECK(arr[1].value() == arr[1].value());
            CHECK(arr[1].value() == static_cast<struct_value>(arr[1].value()));

            // Different elements of the same array: different records.
            CHECK_FALSE(arr[0].value() == arr[1].value());

            // The same values in another array are equal.
            struct_array other = make();
            CHECK(arr[1].value() == other[1].value());
            CHECK(arr[1].value() == static_cast<struct_value>(other[1].value()));
        }

        SUBCASE("assignment through the reference")
        {
            struct_array arr = make();

            // From a struct_value.
            arr[0].value() = static_cast<struct_value>(arr[2].value());
            CHECK_NULLABLE_VARIANT_EQ(arr[0].value()[0], std::int32_t(2));
            CHECK_NULLABLE_VARIANT_EQ(arr[0].value()[1], std::uint8_t(2));

            // From another reference (copy assignment).
            arr[1].value() = arr[0].value();
            CHECK_NULLABLE_VARIANT_EQ(arr[1].value()[0], std::int32_t(2));

            // Assigning an element to itself through two references leaves it unchanged.
            {
                auto lhs = arr[1].value();
                auto rhs = arr[1].value();
                lhs = rhs;
                CHECK_NULLABLE_VARIANT_EQ(arr[1].value()[0], std::int32_t(2));
            }

            // Move assignment: the destination element takes the source record.
            auto target = arr[0].value();
            auto source = arr[2].value();
            target = std::move(source);
            CHECK_NULLABLE_VARIANT_EQ(arr[0].value()[0], std::int32_t(2));
        }

        SUBCASE("the reference reads the element it was created from")
        {
            struct_array arr = make();
            auto reference = arr[1].value();

            // Assigning through the reference updates both the array and the reference itself.
            reference = static_cast<struct_value>(arr[2].value());
            CHECK_NULLABLE_VARIANT_EQ(reference[0], std::int32_t(2));
            CHECK_NULLABLE_VARIANT_EQ(arr[1].value()[0], std::int32_t(2));

            // The reference is index based: it keeps denoting the element at that index.
            CHECK_NULLABLE_VARIANT_EQ(reference[1], std::uint8_t(2));
            CHECK_EQ(reference.size(), 2);
        }
    }

    TEST_CASE_TEMPLATE("struct[T, uint8]", T, std::uint8_t, std::int32_t, float32_t, float64_t)
    {
        using inner_scalar_type = T;
        // using inner_nullable_type = nullable<inner_scalar_type>;

        // number of elements in the struct array
        const std::size_t n = 4;
        const std::size_t n2 = 3;

        // create a struct array
        arrow_proxy proxy = test::make_struct_proxy<inner_scalar_type, uint8_t>(n);
        struct_array struct_arr(std::move(proxy));
        REQUIRE(struct_arr.size() == n);

        SUBCASE("copy")
        {
#ifdef SPARROW_TRACK_COPIES
            copy_tracker::reset(copy_tracker::key<struct_array>());
#endif
            struct_array struct_arr2(struct_arr);
            CHECK_EQ(struct_arr, struct_arr2);
#ifdef SPARROW_TRACK_COPIES
            CHECK_EQ(copy_tracker::count(copy_tracker::key<struct_array>()), 1);
#endif

            struct_array struct_arr3(test::make_struct_proxy<inner_scalar_type, uint8_t>(n2));
            CHECK_NE(struct_arr3, struct_arr);
            struct_arr3 = struct_arr;
            CHECK_EQ(struct_arr3, struct_arr);
        }

        SUBCASE("move")
        {
            struct_array struct_arr2(struct_arr);
            struct_array struct_arr3(std::move(struct_arr2));
            CHECK_EQ(struct_arr3, struct_arr);

            struct_array struct_arr4(test::make_struct_proxy<inner_scalar_type, uint8_t>(n2));
            CHECK_NE(struct_arr4, struct_arr);
            struct_arr4 = std::move(struct_arr3);
            CHECK_EQ(struct_arr4, struct_arr);
        }

        SUBCASE("operator[]")
        {
            for (std::size_t i = 0; i < n; ++i)
            {
                auto val = struct_arr[i];
                REQUIRE(val.has_value());
                auto struct_val = val.value();
                REQUIRE_EQ(struct_val.size(), 2);

                auto val0_variant = struct_val[0];
                auto val1_variant = struct_val[1];

                REQUIRE(val0_variant.has_value());
                REQUIRE(val1_variant.has_value());

                // using const_scalar_ref = const inner_scalar_type&;
                using nullable_inner_scalar_type = nullable<const inner_scalar_type&, bool>;
                using nullable_uint8_t = nullable<const std::uint8_t&, bool>;

#if SPARROW_GCC_11_2_WORKAROUND
                using variant_type = std::decay_t<decltype(val0_variant)>;
                using base_type = typename variant_type::base_type;
#endif
                // visit the variant
                std::visit(
                    [&i](auto&& val0)
                    {
                        if constexpr (std::is_same_v<std::decay_t<decltype(val0)>, nullable_inner_scalar_type>)
                        {
                            CHECK_EQ(val0.value(), static_cast<inner_scalar_type>(i));
                        }
                        else
                        {
                            FAIL("unexpected type");
                        }
                    },
#if SPARROW_GCC_11_2_WORKAROUND
                    static_cast<const base_type&>(val0_variant)
#else
                    val0_variant
#endif
                );

                std::visit(
                    [&i](auto&& val1)
                    {
                        if constexpr (std::is_same_v<std::decay_t<decltype(val1)>, nullable_uint8_t>)
                        {
                            CHECK_EQ(val1.value(), static_cast<inner_scalar_type>(i));
                        }
                        else
                        {
                            FAIL("unexpected type");
                        }
                    },
#if SPARROW_GCC_11_2_WORKAROUND
                    static_cast<const base_type&>(val1_variant)
#else
                    val1_variant
#endif
                );
            }
        }

        SUBCASE("at")
        {
            for (std::size_t i = 0; i < n; ++i)
            {
                auto val = struct_arr[i];
                REQUIRE(val.has_value());
                auto struct_val = val.value();
                REQUIRE_EQ(struct_val.size(), 2);

                auto val0_variant = struct_val.at(0);
                auto val1_variant = struct_val.at(1);

                REQUIRE(val0_variant.has_value());
                REQUIRE(val1_variant.has_value());

                // using const_scalar_ref = const inner_scalar_type&;
                using nullable_inner_scalar_type = nullable<const inner_scalar_type&, bool>;
                using nullable_uint8_t = nullable<const std::uint8_t&, bool>;

#if SPARROW_GCC_11_2_WORKAROUND
                using variant_type = std::decay_t<decltype(val0_variant)>;
                using base_type = typename variant_type::base_type;
#endif
                // visit the variant
                std::visit(
                    [&i](auto&& val0)
                    {
                        if constexpr (std::is_same_v<std::decay_t<decltype(val0)>, nullable_inner_scalar_type>)
                        {
                            CHECK_EQ(val0.value(), static_cast<inner_scalar_type>(i));
                        }
                        else
                        {
                            FAIL("unexpected type");
                        }
                    },
#if SPARROW_GCC_11_2_WORKAROUND
                    static_cast<const base_type&>(val0_variant)
#else
                    val0_variant
#endif
                );

                std::visit(
                    [&i](auto&& val1)
                    {
                        if constexpr (std::is_same_v<std::decay_t<decltype(val1)>, nullable_uint8_t>)
                        {
                            CHECK_EQ(val1.value(), static_cast<inner_scalar_type>(i));
                        }
                        else
                        {
                            FAIL("unexpected type");
                        }
                    },
#if SPARROW_GCC_11_2_WORKAROUND
                    static_cast<const base_type&>(val1_variant)
#else
                    val1_variant
#endif
                );
            }
        }

        SUBCASE("at outside bounds")
        {
            auto val = struct_arr[0];
            REQUIRE(val.has_value());
            auto struct_val = val.value();
            REQUIRE_EQ(struct_val.size(), 2);
            CHECK_THROWS_AS(struct_val.at(2), std::out_of_range);
            CHECK_THROWS_AS(struct_val.at(100), std::out_of_range);
        }

        SUBCASE("operator==(struct_value, struct_value)")
        {
            CHECK(struct_arr[0] == struct_arr[0]);
            CHECK(struct_arr[0] != struct_arr[1]);
        }

        SUBCASE("consistency")
        {
            test::generic_consistency_test(struct_arr);
        }

        SUBCASE("add_child")
        {
            primitive_array<std::int16_t> new_child(
                {{std::int16_t(90), std::int16_t(91), std::int16_t(92), std::int16_t(93)}, true, "new_child"}
            );
            struct_arr.add_child(std::move(new_child));
            CHECK_EQ(struct_arr.children_count(), 3);
            CHECK_EQ(struct_arr.names().back(), "new_child");
            CHECK_NULLABLE_VARIANT_EQ(struct_arr[0].value().at(2), std::int16_t(90));
        }

        SUBCASE("set_child")
        {
            primitive_array<std::int16_t> new_child(
                {{std::int16_t(90), std::int16_t(91), std::int16_t(92), std::int16_t(93)}, true, "new_child"}
            );
            struct_arr.set_child(std::move(new_child), 1);
            CHECK_EQ(struct_arr.children_count(), 2);
            CHECK_EQ(struct_arr.names().back(), "new_child");
            CHECK_NULLABLE_VARIANT_EQ(struct_arr[0].value().at(1), std::int16_t(90));
        }

        SUBCASE("pop_children")
        {
            struct_arr.pop_children(1);
            CHECK_EQ(struct_arr.children_count(), 1);
        }

#if defined(__cpp_lib_format)
        SUBCASE("formatting")
        {
            const std::string formatted = std::format("{}", struct_arr);
            constexpr std::string_view expected = "|item 0|item 1|\n"
                                                  "---------------\n"
                                                  "|     0|     0|\n"
                                                  "|     1|     1|\n"
                                                  "|     2|     2|\n"
                                                  "|     3|     3|\n"
                                                  "---------------";
            CHECK_EQ(formatted, expected);

            // operator<< goes through the same formatter.
            std::ostringstream stream;
            stream << struct_arr;
            CHECK_EQ(stream.str(), formatted);

            // A null element formats as an empty cell (ref.has_value() == false).
            struct_array with_null = struct_arr;
            with_null[1] = nullval;
            const std::string with_null_formatted = std::format("{}", with_null);
            CHECK_NE(with_null_formatted, formatted);
        }
#endif

        SUBCASE("struct_value iterators")
        {
            // Get a struct_value from the array
            auto val = struct_arr[0];
            REQUIRE(val.has_value());
            auto struct_val = val.value();
            REQUIRE_EQ(struct_val.size(), 2);

            // begin() and end() iteration
            int count1 = 0;
            for (auto it = struct_val.begin(); it != struct_val.end(); ++it)
            {
                CHECK(it->has_value());
                ++count1;
            }
            CHECK_EQ(count1, 2);

            // cbegin() and cend() iteration
            int count2 = 0;
            for (auto it = struct_val.cbegin(); it != struct_val.cend(); ++it)
            {
                CHECK(it->has_value());
                ++count2;
            }
            CHECK_EQ(count2, 2);

            // Range-based for loop
            int count3 = 0;
            for (const auto& elem : struct_val)
            {
                CHECK(elem.has_value());
                ++count3;
            }
            CHECK_EQ(count3, 2);
        }

        SUBCASE("struct_value begin/end iteration")
        {
            auto val = struct_arr[2];
            REQUIRE(val.has_value());
            auto struct_val = val.value();
            REQUIRE_EQ(struct_val.size(), 2);
            int i = 0;
            for (auto it = struct_val.begin(); it != struct_val.end(); ++it, ++i)
            {
                if (i == 0)
                {
                    CHECK_NULLABLE_VARIANT_EQ(*it, static_cast<inner_scalar_type>(2));
                }
                else if (i == 1)
                {
                    CHECK_NULLABLE_VARIANT_EQ(*it, static_cast<std::uint8_t>(2));
                }
            }
            CHECK_EQ(i, 2);
        }

        SUBCASE("struct_value cbegin/cend iteration")
        {
            auto val = struct_arr[2];
            REQUIRE(val.has_value());
            auto struct_val = val.value();
            REQUIRE_EQ(struct_val.size(), 2);
            int i = 0;
            for (auto it = struct_val.cbegin(); it != struct_val.cend(); ++it, ++i)
            {
                if (i == 0)
                {
                    CHECK_NULLABLE_VARIANT_EQ(*it, static_cast<inner_scalar_type>(2));
                }
                else if (i == 1)
                {
                    CHECK_NULLABLE_VARIANT_EQ(*it, static_cast<std::uint8_t>(2));
                }
            }
            CHECK_EQ(i, 2);
        }

        SUBCASE("struct_value range-based for loop")
        {
            auto val = struct_arr[2];
            REQUIRE(val.has_value());
            auto struct_val = val.value();
            REQUIRE_EQ(struct_val.size(), 2);
            int i = 0;
            for (const auto& elem : struct_val)
            {
                if (i == 0)
                {
                    CHECK_NULLABLE_VARIANT_EQ(elem, static_cast<inner_scalar_type>(2));
                }
                else if (i == 1)
                {
                    CHECK_NULLABLE_VARIANT_EQ(elem, static_cast<std::uint8_t>(2));
                }
                ++i;
            }
            CHECK_EQ(i, 2);
        }
    }
}
