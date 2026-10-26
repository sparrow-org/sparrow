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

#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>

#include "sparrow/array.hpp"
#include "sparrow/array_api.hpp"
#include "sparrow/arrow_interface/arrow_array_schema_proxy.hpp"
#include "sparrow/dictionary_encoded_array.hpp"
#include "sparrow/types/data_traits.hpp"
#include "sparrow/types/data_type.hpp"
#include "sparrow/utils/nullable.hpp"
#include "sparrow/variable_size_binary_array.hpp"

#include "doctest/doctest.h"
#include "metadata_sample.hpp"
#include "test_utils.hpp"

namespace sparrow
{
    using keys_type = uint32_t;
    using layout_type = dictionary_encoded_array<keys_type>;
    using layout_type_cref = layout_type::const_reference;

    static const std::array<nullable<std::string>, 7>
        words{"hello", "you", {"are", false}, "not", "prepared", "!", "?"};

    inline layout_type make_dictionary()
    {
        layout_type::keys_buffer_type keys{0, 0, 1, 2, 3, 4, 2, 5, 0, 1, 2};

        string_array words_arr{words};
        array ar(words_arr.slice(1, words_arr.size()));

        constexpr std::array<size_t, 2> keys_nulls{1ULL, 5ULL};
        // Indexes: 0(null), 1, 2, 3, 4(null), 2, 5, 0, 1, 2

        //// Values: you, are(null), not, prepared, !, ?

        // null, null, not, prepared, null, not, ?, you, are(null), not
        const layout_type dict{std::move(keys), std::move(ar), std::move(keys_nulls), "name", metadata_sample_opt};
        return dict.slice(1, dict.size());
    }

    nullable<std::string_view> get_dict_value(layout_type_cref r)
    {
        return std::get<nullable<std::string_view>>(r);
    }

    inline layout_type make_mutable_dictionary()
    {
        layout_type::keys_buffer_type keys{0, 1, 2, 0};
        std::vector<nullable<std::string>> values_words{"a", "b", "c"};
        string_array values{values_words};
        array values_array{std::move(values)};
        return layout_type{std::move(keys), std::move(values_array), true};
    }

    template <class T>
    std::size_t dictionary_values_size(const T& dict)
    {
        return static_cast<std::size_t>(detail::array_access::get_arrow_proxy(dict).dictionary()->length());
    }

    TEST_SUITE("dictionary_encoded_array")
    {
        TEST_CASE("constructors")
        {
            CHECK_NOTHROW(make_dictionary());

            SUBCASE("keys_buffer_type, array, validity_bitmap, name and metadata")
            {
                layout_type::keys_buffer_type keys{0, 0, 1, 2, 3, 4, 2, 5, 0, 1, 2};
                string_array words_arr{words};
                array ar(words_arr.slice(1, words_arr.size()));
                constexpr std::array<size_t, 2> keys_nulls{1ULL, 5ULL};
                const layout_type
                    dict{std::move(keys), std::move(ar), std::move(keys_nulls), "name", metadata_sample_opt};
                CHECK_EQ(dict.size(), 11);
            }

            SUBCASE("keys_buffer_type, array, nullable, name and metadata")
            {
                layout_type::keys_buffer_type keys{0, 0, 1, 2, 3, 4, 2, 5, 0, 1, 2};
                string_array words_arr{words};
                array ar(words_arr.slice(1, words_arr.size()));
                constexpr std::array<size_t, 2> keys_nulls{1ULL, 5ULL};
                const layout_type
                    dict{std::move(keys), std::move(ar), std::move(keys_nulls), "name", metadata_sample_opt};
                CHECK_EQ(dict.size(), 11);
            }

            SUBCASE("KEY_RANGE, array, bitmap, name and metadata")
            {
                std::vector<uint32_t> keys{0, 0, 1, 2, 3, 4, 2, 5, 0, 1, 2};
                string_array words_arr{words};
                array ar(words_arr.slice(1, words_arr.size()));
                constexpr std::array<size_t, 2> keys_nulls{1ULL, 5ULL};
                const layout_type
                    dict{std::move(keys), std::move(ar), std::move(keys_nulls), "name", metadata_sample_opt};
                CHECK_EQ(dict.size(), 11);
            }

            SUBCASE("nullable key range, name and metadata")
            {
                std::vector<nullable<keys_type>> keys{
                    sparrow::make_nullable<keys_type>(0, true),
                    sparrow::make_nullable<keys_type>(0, false),
                    sparrow::make_nullable<keys_type>(1, true),
                    sparrow::make_nullable<keys_type>(2, true),
                    sparrow::make_nullable<keys_type>(3, true),
                    sparrow::make_nullable<keys_type>(4, false),
                    sparrow::make_nullable<keys_type>(2, true),
                    sparrow::make_nullable<keys_type>(5, true),
                    sparrow::make_nullable<keys_type>(0, true),
                    sparrow::make_nullable<keys_type>(1, true),
                    sparrow::make_nullable<keys_type>(2, true)
                };
                string_array words_arr{words};
                array ar(words_arr.slice(1, words_arr.size()));
                const layout_type dict{std::move(keys), std::move(ar), "name", metadata_sample_opt};
                CHECK_EQ(dict.size(), 11);
            }
        }

        TEST_CASE("convenience_constructors")
        {
            using key_type = std::uint32_t;
            using array_type = dictionary_encoded_array<key_type>;
            using keys_buffer_type = typename array_type::keys_buffer_type;

            // the value array
            primitive_array<float32_t> values{{0.0f, 1.0f, 2.0f, 3.0f}};

            // detyped array
            array values_arr(std::move(values));

            // the keys **data**
            keys_buffer_type keys{3, 3, 2, 1, 0};

            // where nulls are
            std::vector<std::size_t> where_null{2};

            // create the array
            auto arr = array_type(
                std::move(keys),
                std::move(values_arr),
                std::move(where_null),
                "name",
                metadata_sample_opt
            );

            // check the size
            REQUIRE_EQ(arr.size(), 5);

            CHECK_EQ(arr.name(), "name");
            test_metadata(metadata_sample, *(arr.metadata()));

            // check bitmap
            REQUIRE_EQ(std::as_const(arr)[0].has_value(), true);
            REQUIRE_EQ(std::as_const(arr)[1].has_value(), true);
            REQUIRE_EQ(std::as_const(arr)[2].has_value(), false);
            REQUIRE_EQ(std::as_const(arr)[3].has_value(), true);
            REQUIRE_EQ(std::as_const(arr)[4].has_value(), true);

            // check the values
            CHECK_NULLABLE_VARIANT_EQ(std::as_const(arr)[0], float32_t(3.0f));
            CHECK_NULLABLE_VARIANT_EQ(std::as_const(arr)[1], float32_t(3.0f));
            CHECK_NULLABLE_VARIANT_EQ(std::as_const(arr)[3], float32_t(1.0f));
            CHECK_NULLABLE_VARIANT_EQ(std::as_const(arr)[4], float32_t(0.0f));
        }

        TEST_CASE("copy")
        {
#ifdef SPARROW_TRACK_COPIES
            copy_tracker::reset(copy_tracker::key<layout_type>());
#endif
            layout_type ar = make_dictionary();
            layout_type ar2(ar);
            CHECK_EQ(ar, ar2);
#ifdef SPARROW_TRACK_COPIES
            CHECK_EQ(copy_tracker::count(copy_tracker::key<layout_type>()), 1);
#endif

            layout_type ar3 = make_dictionary();
            ar3 = ar;
            CHECK_EQ(ar, ar3);
        }

        TEST_CASE("move")
        {
            layout_type ar = make_dictionary();
            layout_type ar2(ar);
            layout_type ar3(std::move(ar));
            CHECK_EQ(ar2, ar3);

            layout_type ar4 = make_dictionary();
            ar4 = std::move(ar3);
            CHECK_EQ(ar2, ar4);
        }

        TEST_CASE("size")
        {
            const layout_type dict = make_dictionary();
            CHECK_EQ(dict.size(), 10);
        }

        TEST_CASE("operator[]")
        {
            const layout_type dict = make_dictionary();
            CHECK_FALSE(dict[0].has_value());
            CHECK_FALSE(dict[1].has_value());
            REQUIRE(dict[2].has_value());
            CHECK_EQ(get_dict_value(dict[2]).value(), words[3]);
            REQUIRE(dict[3].has_value());
            CHECK_EQ(get_dict_value(dict[3]).value(), words[4]);
            CHECK_FALSE(dict[4].has_value());
            REQUIRE(dict[5].has_value());
            CHECK_EQ(get_dict_value(dict[5]).value(), words[3]);
            REQUIRE(dict[6].has_value());
            CHECK_EQ(get_dict_value(dict[6]).value(), words[6]);
            REQUIRE(dict[7].has_value());
            CHECK_EQ(get_dict_value(dict[7]).value(), words[1]);
            CHECK_FALSE(dict[8].has_value());
            REQUIRE(dict[9].has_value());
            CHECK_EQ(get_dict_value(dict[9]).value(), words[3]);
        }

        TEST_CASE("const_iterator")
        {
            const layout_type dict = make_dictionary();
            auto iter = dict.cbegin();
            CHECK_EQ(*iter, dict[0]);
            ++iter;
            CHECK_EQ(*iter, dict[1]);
            ++iter;
            CHECK_EQ(*iter, dict[2]);
            ++iter;
            --iter;
            CHECK_EQ(*iter, dict[2]);
            ++iter;
            CHECK_EQ(*iter, dict[3]);
            iter++;
            CHECK_EQ(*iter, dict[4]);
            ++iter;
            CHECK_EQ(*iter, dict[5]);
            ++iter;
            CHECK_EQ(*iter, dict[6]);
            ++iter;
            CHECK_EQ(*iter, dict[7]);
            iter += 3;
            CHECK_EQ(iter, dict.cend());
        }

        TEST_CASE("reverse_iterator")
        {
            const layout_type dict = make_dictionary();
            auto iter = dict.crbegin();
            CHECK_EQ(*iter, dict[9]);
            ++iter;
            CHECK_EQ(*iter, dict[8]);
            ++iter;
            CHECK_EQ(*iter, dict[7]);
            ++iter;
            --iter;
            CHECK_EQ(*iter, dict[7]);
            ++iter;
            CHECK_EQ(*iter, dict[6]);
            iter++;
            CHECK_EQ(*iter, dict[5]);
            ++iter;
            CHECK_EQ(*iter, dict[4]);
            ++iter;
            CHECK_EQ(*iter, dict[3]);
        }

        TEST_CASE("empty")
        {
            const layout_type dict = make_dictionary();
            CHECK_FALSE(dict.empty());

            // Test with empty dictionary
            layout_type::keys_buffer_type empty_keys{};
            string_array empty_words_arr{std::vector<nullable<std::string>>{}};
            array empty_ar(std::move(empty_words_arr));
            const layout_type
                empty_dict{std::move(empty_keys), std::move(empty_ar), false, "empty", metadata_sample_opt};
            CHECK(empty_dict.empty());
            CHECK_EQ(empty_dict.size(), 0);
        }

        TEST_CASE("front")
        {
            const layout_type dict = make_dictionary();
            CHECK_FALSE(dict.front().has_value());
            CHECK_EQ(dict.front(), dict[0]);
        }

        TEST_CASE("back")
        {
            const layout_type dict = make_dictionary();
            REQUIRE(dict.back().has_value());
            CHECK_EQ(get_dict_value(dict.back()).value(), words[3]);
            CHECK_EQ(dict.back(), dict[9]);
        }

        TEST_CASE("slice")
        {
            const layout_type dict = make_dictionary();
            auto sliced = dict.slice(2, 6);
            CHECK_EQ(sliced.size(), 4);
            // Test that slice creates a valid array with the correct size
            // and that elements can be accessed without errors
            for (size_t i = 0; i < sliced.size(); ++i)
            {
                // Just ensure we can access elements without crashing
                [[maybe_unused]] auto element = sliced[i];
            }
        }

        TEST_CASE("slice_view")
        {
            const layout_type dict = make_dictionary();
            auto sliced_view = dict.slice_view(1, 5);
            CHECK_EQ(sliced_view.size(), 4);
            // Test that slice_view creates a valid array with the correct size
            // and that elements can be accessed without errors
            for (size_t i = 0; i < sliced_view.size(); ++i)
            {
                // Just ensure we can access elements without crashing
                [[maybe_unused]] auto element = std::as_const(sliced_view)[i];
            }
        }

        TEST_CASE("slice_inplace")
        {
            auto dict = make_dictionary();
            dict.slice_inplace(2, 6);
            CHECK_EQ(dict.size(), 4);
            CHECK_EQ(dict.offset(), 3);  // initial offset=1 (from make_dictionary's slice), +2 = 3

            dict.slice_inplace(1, 3);
            CHECK_EQ(dict.size(), 2);
            CHECK_EQ(dict.offset(), 4);  // previous offset=3, +1 = 4

            for (size_t i = 0; i < dict.size(); ++i)
            {
                [[maybe_unused]] auto element = std::as_const(dict)[i];
            }
        }

        TEST_CASE("name_and_metadata")
        {
            const layout_type dict = make_dictionary();
            CHECK_EQ(dict.name().value(), "name");

            // Test metadata separately with a simple case
            layout_type::keys_buffer_type keys{0, 1};
            string_array words_arr{std::vector<nullable<std::string>>{"a", "b"}};
            array ar(std::move(words_arr));

            // Create metadata manually
            std::optional<std::vector<std::pair<std::string, std::string>>> simple_metadata{
                std::vector<std::pair<std::string, std::string>>{{"test_key", "test_value"}}
            };
            const layout_type simple_dict{std::move(keys), std::move(ar), false, "test", simple_metadata};

            CHECK_EQ(simple_dict.name().value(), "test");
            CHECK(simple_dict.metadata().has_value());
        }

        TEST_CASE("equality")
        {
            const layout_type dict1 = make_dictionary();
            const layout_type dict2 = make_dictionary();
            CHECK_EQ(dict1, dict2);

            // Test with different data
            layout_type::keys_buffer_type keys{0, 1, 2};
            string_array words_arr{std::vector<nullable<std::string>>{"a", "b", "c"}};
            array ar(std::move(words_arr));
            const layout_type dict3{std::move(keys), std::move(ar), false, "different", metadata_sample_opt};
            CHECK_NE(dict1, dict3);
        }

        TEST_CASE("non_mutable_iterators")
        {
            const layout_type dict = make_dictionary();

            // Test that begin() on const object returns const_iterator
            auto iter = dict.begin();
            CHECK_EQ(iter, dict.cbegin());

            // Test that end() on const object returns const_iterator
            auto end_iter = dict.end();
            CHECK_EQ(end_iter, dict.cend());
        }

        TEST_CASE("iterator_arithmetic")
        {
            const layout_type dict = make_dictionary();
            auto iter = dict.cbegin();

            // Test += operator
            iter += 3;
            CHECK_EQ(*iter, dict[3]);

            // Test -= operator (if available)
            iter -= 2;
            CHECK_EQ(*iter, dict[1]);

            // Test iterator difference
            auto iter2 = dict.cbegin();
            auto diff = iter - iter2;
            CHECK_EQ(diff, 1);
        }

        TEST_CASE("range_based_for_loop")
        {
            const layout_type dict = make_dictionary();
            size_t index = 0;
            for (const auto& element : dict)
            {
                CHECK_EQ(element, dict[index]);
                ++index;
            }
            CHECK_EQ(index, dict.size());
        }

        TEST_CASE("edge_cases")
        {
            SUBCASE("single_element")
            {
                layout_type::keys_buffer_type keys{0};
                string_array words_arr{std::vector<nullable<std::string>>{"single"}};
                array ar(std::move(words_arr));
                const layout_type single_dict{std::move(keys), std::move(ar), false, "single", metadata_sample_opt};

                CHECK_EQ(single_dict.size(), 1);
                CHECK_EQ(single_dict.front(), single_dict.back());
                REQUIRE(single_dict[0].has_value());
                CHECK_EQ(std::get<nullable<std::string_view>>(single_dict[0]).value(), "single");
            }

            SUBCASE("all_nulls")
            {
                layout_type::keys_buffer_type keys{0, 1, 2};
                string_array words_arr{std::vector<nullable<std::string>>{"a", "b", "c"}};
                array ar(std::move(words_arr));
                std::vector<size_t> all_nulls{0, 1, 2};
                const layout_type null_dict{
                    std::move(keys),
                    std::move(ar),
                    std::move(all_nulls),
                    "nulls",
                    metadata_sample_opt
                };

                CHECK_EQ(null_dict.size(), 3);
                CHECK_FALSE(null_dict[0].has_value());
                CHECK_FALSE(null_dict[1].has_value());
                CHECK_FALSE(null_dict[2].has_value());
            }

            SUBCASE("no_nulls")
            {
                layout_type::keys_buffer_type keys{0, 1, 2, 0, 1};
                string_array words_arr{std::vector<nullable<std::string>>{"a", "b", "c"}};
                array ar(std::move(words_arr));
                const layout_type
                    no_null_dict{std::move(keys), std::move(ar), false, "no_nulls", metadata_sample_opt};

                CHECK_EQ(no_null_dict.size(), 5);
                for (size_t i = 0; i < no_null_dict.size(); ++i)
                {
                    REQUIRE(no_null_dict[i].has_value());
                }
            }
        }


        TEST_CASE("set_value")
        {
            SUBCASE("new value appends to the dictionary")
            {
                layout_type dict = make_mutable_dictionary();
                dict.set_value(3, layout_type::value_type{make_nullable<std::string>("d")});
                REQUIRE_EQ(dict.size(), 4);
                CHECK_EQ(dictionary_values_size(dict), 4);
                CHECK_NULLABLE_VARIANT_EQ(std::as_const(dict)[3], std::string_view("d"));
            }

            SUBCASE("existing value reuses the dictionary entry")
            {
                layout_type dict = make_mutable_dictionary();
                dict.set_value(3, make_nullable<std::string>("c"));
                REQUIRE_EQ(dict.size(), 4);
                CHECK_EQ(dictionary_values_size(dict), 3);
                CHECK_NULLABLE_VARIANT_EQ(std::as_const(dict)[2], std::string_view("c"));
                CHECK_NULLABLE_VARIANT_EQ(std::as_const(dict)[3], std::string_view("c"));
            }

            SUBCASE("replacement compacts orphaned entries")
            {
                layout_type dict = make_mutable_dictionary();
                dict.set_value(1, make_nullable<std::string>("a"));
                REQUIRE_EQ(dict.size(), 4);
                CHECK_EQ(dictionary_values_size(dict), 2);
                const auto& cdict = dict;
                CHECK_NULLABLE_VARIANT_EQ(cdict[0], std::string_view("a"));
                CHECK_NULLABLE_VARIANT_EQ(cdict[1], std::string_view("a"));
                CHECK_NULLABLE_VARIANT_EQ(cdict[2], std::string_view("c"));
                CHECK_NULLABLE_VARIANT_EQ(cdict[3], std::string_view("a"));
            }

            SUBCASE("null replacement compacts orphaned entries")
            {
                layout_type dict = make_mutable_dictionary();
                dict.set_value(1, sparrow::nullval);
                REQUIRE_EQ(dict.size(), 4);
                CHECK_FALSE(std::as_const(dict)[1].has_value());
                CHECK_EQ(dictionary_values_size(dict), 2);
                CHECK_NULLABLE_VARIANT_EQ(std::as_const(dict)[2], std::string_view("c"));
                CHECK_NULLABLE_VARIANT_EQ(std::as_const(dict)[3], std::string_view("a"));
            }

            SUBCASE("null replacement does not append the value")
            {
                layout_type dict = make_mutable_dictionary();
                dict.set_value(0, make_nullable<std::string>("x", false));
                CHECK_FALSE(std::as_const(dict)[0].has_value());
                CHECK_EQ(dictionary_values_size(dict), 3);
            }

            SUBCASE("null dictionary entries never match a lookup")
            {
                std::vector<nullable<std::string>> dictionary_values{
                    make_nullable<std::string>("a"),
                    nullable<std::string>{},
                    make_nullable<std::string>("b")
                };
                string_array values{dictionary_values};
                array values_array{std::move(values)};
                layout_type dict{layout_type::keys_buffer_type{0, 1, 2}, std::move(values_array), true};
                CHECK_FALSE(std::as_const(dict)[1].has_value());

                dict.set_value(1, make_nullable<std::string>("b"));
                CHECK_EQ(dictionary_values_size(dict), 2);
                CHECK_NULLABLE_VARIANT_EQ(std::as_const(dict)[1], std::string_view("b"));
                CHECK_NULLABLE_VARIANT_EQ(std::as_const(dict)[2], std::string_view("b"));
            }
        }

        TEST_CASE("insert")
        {
            SUBCASE("push_back appends a new value once")
            {
                layout_type dict = make_mutable_dictionary();
                dict.push_back(make_nullable<std::string>("d"));
                CHECK_EQ(dict.size(), 5);
                CHECK_EQ(dictionary_values_size(dict), 4);
                dict.push_back(make_nullable<std::string>("d"));
                CHECK_EQ(dict.size(), 6);
                CHECK_EQ(dictionary_values_size(dict), 4);
                CHECK_NULLABLE_VARIANT_EQ(std::as_const(dict)[5], std::string_view("d"));
            }

            SUBCASE("push_back reuses an existing value")
            {
                layout_type dict = make_mutable_dictionary();
                dict.push_back(make_nullable<std::string>("c"));
                CHECK_EQ(dict.size(), 5);
                CHECK_EQ(dictionary_values_size(dict), 3);
                CHECK_NULLABLE_VARIANT_EQ(std::as_const(dict)[4], std::string_view("c"));
            }

            SUBCASE("push_back null keeps the dictionary unchanged")
            {
                layout_type dict = make_mutable_dictionary();
                dict.push_back(nullable<std::string>{});
                CHECK_EQ(dict.size(), 5);
                CHECK_FALSE(std::as_const(dict)[4].has_value());
                CHECK_EQ(dictionary_values_size(dict), 3);
            }

            SUBCASE("insert with a count")
            {
                layout_type dict = make_mutable_dictionary();
                dict.insert(dict.cbegin() + 1, make_nullable<std::string>("d"), 2);
                REQUIRE_EQ(dict.size(), 6);
                CHECK_EQ(dictionary_values_size(dict), 4);
                const auto& cdict = dict;
                CHECK_NULLABLE_VARIANT_EQ(cdict[0], std::string_view("a"));
                CHECK_NULLABLE_VARIANT_EQ(cdict[1], std::string_view("d"));
                CHECK_NULLABLE_VARIANT_EQ(cdict[2], std::string_view("d"));
                CHECK_NULLABLE_VARIANT_EQ(cdict[3], std::string_view("b"));
            }

            SUBCASE("range insert")
            {
                layout_type dict = make_mutable_dictionary();
                const std::vector<nullable<std::string>> values{
                    make_nullable<std::string>("b"),
                    make_nullable<std::string>("e"),
                    make_nullable<std::string>("a")
                };
                dict.insert(dict.cbegin(), values);
                REQUIRE_EQ(dict.size(), 7);
                CHECK_EQ(dictionary_values_size(dict), 4);
                const auto& cdict = dict;
                CHECK_NULLABLE_VARIANT_EQ(cdict[0], std::string_view("b"));
                CHECK_NULLABLE_VARIANT_EQ(cdict[1], std::string_view("e"));
                CHECK_NULLABLE_VARIANT_EQ(cdict[2], std::string_view("a"));
                CHECK_NULLABLE_VARIANT_EQ(cdict[3], std::string_view("a"));
            }

            SUBCASE("self range insert")
            {
                layout_type dict = make_mutable_dictionary();
                dict.insert(dict.cend(), dict.cbegin(), dict.cend());
                REQUIRE_EQ(dict.size(), 8);
                CHECK_EQ(dictionary_values_size(dict), 3);
                const auto& cdict = dict;
                const std::array<std::string_view, 8> expected{"a", "b", "c", "a", "a", "b", "c", "a"};
                for (std::size_t i = 0; i < expected.size(); ++i)
                {
                    CHECK_NULLABLE_VARIANT_EQ(cdict[i], expected[i]);
                }
            }
        }

        TEST_CASE("erase")
        {
            SUBCASE("erase compacts the dictionary")
            {
                layout_type dict = make_mutable_dictionary();
                dict.erase(dict.cbegin() + 1);
                REQUIRE_EQ(dict.size(), 3);
                CHECK_EQ(dictionary_values_size(dict), 2);
                const auto& cdict = dict;
                CHECK_NULLABLE_VARIANT_EQ(cdict[0], std::string_view("a"));
                CHECK_NULLABLE_VARIANT_EQ(cdict[1], std::string_view("c"));
                CHECK_NULLABLE_VARIANT_EQ(cdict[2], std::string_view("a"));
            }

            SUBCASE("range erase")
            {
                layout_type dict = make_mutable_dictionary();
                dict.erase(dict.cbegin(), dict.cbegin() + 2);
                REQUIRE_EQ(dict.size(), 2);
                CHECK_EQ(dictionary_values_size(dict), 2);
                const auto& cdict = dict;
                CHECK_NULLABLE_VARIANT_EQ(cdict[0], std::string_view("c"));
                CHECK_NULLABLE_VARIANT_EQ(cdict[1], std::string_view("a"));
            }

            SUBCASE("pop_back empties the array")
            {
                layout_type dict = make_mutable_dictionary();
                dict.pop_back();
                CHECK_EQ(dict.size(), 3);
                CHECK_EQ(dictionary_values_size(dict), 3);
                dict.pop_back();
                CHECK_EQ(dict.size(), 2);
                CHECK_EQ(dictionary_values_size(dict), 2);
                dict.pop_back();
                CHECK_EQ(dict.size(), 1);
                CHECK_EQ(dictionary_values_size(dict), 1);
                dict.pop_back();
                CHECK(dict.empty());
                CHECK_EQ(dictionary_values_size(dict), 0);
            }
        }

        TEST_CASE("resize")
        {
            SUBCASE("grow with null")
            {
                layout_type dict = make_mutable_dictionary();
                dict.resize(6);
                REQUIRE_EQ(dict.size(), 6);
                CHECK_FALSE(std::as_const(dict)[4].has_value());
                CHECK_FALSE(std::as_const(dict)[5].has_value());
                CHECK_EQ(dictionary_values_size(dict), 3);
            }

            SUBCASE("grow with a value")
            {
                layout_type dict = make_mutable_dictionary();
                dict.resize(6, make_nullable<std::string>("d"));
                REQUIRE_EQ(dict.size(), 6);
                CHECK_EQ(dictionary_values_size(dict), 4);
                CHECK_NULLABLE_VARIANT_EQ(std::as_const(dict)[4], std::string_view("d"));
                CHECK_NULLABLE_VARIANT_EQ(std::as_const(dict)[5], std::string_view("d"));
            }

            SUBCASE("shrink compacts the dictionary")
            {
                layout_type dict = make_mutable_dictionary();
                dict.resize(2);
                REQUIRE_EQ(dict.size(), 2);
                CHECK_EQ(dictionary_values_size(dict), 2);
                const auto& cdict = dict;
                CHECK_NULLABLE_VARIANT_EQ(cdict[0], std::string_view("a"));
                CHECK_NULLABLE_VARIANT_EQ(cdict[1], std::string_view("b"));
            }

            SUBCASE("resize to zero")
            {
                layout_type dict = make_mutable_dictionary();
                dict.resize(0);
                CHECK(dict.empty());
                CHECK_EQ(dictionary_values_size(dict), 0);
            }
        }

        TEST_CASE("mutable_operator[]")
        {
            SUBCASE("assignment appends a new value once")
            {
                layout_type dict = make_mutable_dictionary();
                dict[0] = make_nullable<std::string>("d");
                CHECK_EQ(dictionary_values_size(dict), 4);
                CHECK_NULLABLE_VARIANT_EQ(std::as_const(dict)[0], std::string_view("d"));
                dict[0] = make_nullable<std::string>("d");
                CHECK_EQ(dictionary_values_size(dict), 4);
                CHECK_NULLABLE_VARIANT_EQ(std::as_const(dict)[0], std::string_view("d"));
            }

            SUBCASE("assignment reuses and compacts")
            {
                layout_type dict = make_mutable_dictionary();
                dict[1] = make_nullable<std::string>("a");
                CHECK_EQ(dictionary_values_size(dict), 2);
                CHECK_NULLABLE_VARIANT_EQ(std::as_const(dict)[1], std::string_view("a"));
                CHECK_NULLABLE_VARIANT_EQ(std::as_const(dict)[2], std::string_view("c"));
            }

            SUBCASE("null nullable assignment clears the validity flag")
            {
                layout_type dict = make_mutable_dictionary();
                dict[0] = make_nullable<std::string>("x", false);
                CHECK_FALSE(std::as_const(dict)[0].has_value());
                // "x" is appended by the proxy assignment and stays unreferenced until compaction.
                CHECK_EQ(dictionary_values_size(dict), 4);
                dict.set_value(1, make_nullable<std::string>("b"));
                CHECK_EQ(dictionary_values_size(dict), 3);
            }

            SUBCASE("direct proxy assignment")
            {
                layout_type dict = make_mutable_dictionary();
                dict[3].get() = layout_type::value_type{make_nullable<std::string>("d")};
                CHECK_EQ(dictionary_values_size(dict), 4);
                CHECK_NULLABLE_VARIANT_EQ(std::as_const(dict)[3], std::string_view("d"));

                dict[2].null_flag() = false;
                CHECK_FALSE(std::as_const(dict)[2].has_value());
            }
        }

        TEST_CASE("mutation_edge_cases")
        {
            SUBCASE("sliced arrays cannot be mutated")
            {
                layout_type source = make_mutable_dictionary();
                layout_type dict = source.slice_view(1, 3);
                const layout_type::value_type value{make_nullable<std::string>("x")};
                CHECK_THROWS_AS(dict.set_value(0, value), std::logic_error);
                CHECK_THROWS_AS(dict.set_value(0, sparrow::nullval), std::logic_error);
                // A sliced view has no mutable bitmap, so use a deep copy with a non-zero offset to
                // exercise the proxy assignment path.
                layout_type deep_sliced = source.slice(1, 3);
                CHECK_THROWS_AS(deep_sliced[0] = make_nullable<std::string>("x"), std::logic_error);
                CHECK_THROWS_AS(dict.insert(dict.cbegin(), value), std::logic_error);
                CHECK_THROWS_AS(dict.push_back(value), std::logic_error);
                CHECK_THROWS_AS(dict.erase(dict.cbegin()), std::logic_error);
                CHECK_THROWS_AS(dict.pop_back(), std::logic_error);
                CHECK_THROWS_AS(dict.resize(1), std::logic_error);
                CHECK_THROWS_AS(dict.resize(4, value), std::logic_error);
            }

            SUBCASE("the value type must match the dictionary values")
            {
                primitive_array<std::int32_t> values{{1, 2}};
                layout_type dict{layout_type::keys_buffer_type{0, 1}, array{std::move(values)}, true};
                dict.set_value(0, make_nullable<std::int32_t>(5));
                CHECK_EQ(dictionary_values_size(dict), 2);
                CHECK_NULLABLE_VARIANT_EQ(std::as_const(dict)[0], std::int32_t(5));
                CHECK_NULLABLE_VARIANT_EQ(std::as_const(dict)[1], std::int32_t(2));
                CHECK_THROWS_AS(
                    dict.push_back(make_nullable<std::int64_t>(std::int64_t{7})),
                    std::invalid_argument
                );
                CHECK_THROWS_AS(dict[0] = make_nullable<std::int64_t>(std::int64_t{7}), std::invalid_argument);
                CHECK_EQ(dict.size(), 2);
            }

            SUBCASE("non-nullable arrays only accept non-null values")
            {
                primitive_array<std::int32_t> values{{1, 2}};
                layout_type dict{layout_type::keys_buffer_type{0, 1}, array{std::move(values)}, false};

                CHECK_THROWS_AS(dict[0] = make_nullable<std::int32_t>(5), std::logic_error);
                CHECK_THROWS_AS(dict.set_value(0, sparrow::nullval), std::invalid_argument);
                CHECK_THROWS_AS(dict.push_back(nullable<std::int32_t>{}), std::invalid_argument);
                CHECK_THROWS_AS(dict.resize(3), std::invalid_argument);

                CHECK_NOTHROW(dict.set_value(0, make_nullable<std::int32_t>(5)));
                CHECK_NOTHROW(dict.push_back(make_nullable<std::int32_t>(6)));
                CHECK_EQ(dict.size(), 3);
                CHECK_NULLABLE_VARIANT_EQ(std::as_const(dict)[0], std::int32_t(5));
                CHECK_NULLABLE_VARIANT_EQ(std::as_const(dict)[2], std::int32_t(6));
            }

            SUBCASE("the dictionary index must fit the key type")
            {
                std::vector<nullable<std::string>> dictionary_values;
                dictionary_values.reserve(127);
                for (int i = 0; i < 127; ++i)
                {
                    dictionary_values.push_back(make_nullable<std::string>("v" + std::to_string(i)));
                }
                string_array values{dictionary_values};
                array values_array{std::move(values)};
                using small_layout_type = dictionary_encoded_array<std::int8_t>;
                small_layout_type dict{
                    small_layout_type::keys_buffer_type{std::int8_t{0}},
                    std::move(values_array),
                    true
                };

                dict.push_back(make_nullable<std::string>("new1"));
                CHECK_EQ(dictionary_values_size(dict), 128);
                CHECK_THROWS_AS(dict.push_back(make_nullable<std::string>("new2")), std::overflow_error);
            }
        }

        TEST_CASE("untyped_mutation")
        {
            layout_type mutable_dict = make_mutable_dictionary();
            array arr{std::move(mutable_dict)};
            const array source{make_mutable_dictionary()};

            arr.insert(arr.cend(), source.cbegin(), source.cend());
            REQUIRE_EQ(arr.size(), 8);
            REQUIRE(arr.dictionary().has_value());
            CHECK_EQ(arr.dictionary()->size(), 3);

            arr.erase(arr.cbegin());
            CHECK_EQ(arr.size(), 7);
            REQUIRE(arr.dictionary().has_value());
            CHECK_EQ(arr.dictionary()->size(), 3);
        }

#if defined(__cpp_lib_format)
        TEST_CASE("formatter")
        {
            const layout_type dict = make_dictionary();
            const std::string formatted = std::format("{}", dict);
            constexpr std::string_view
                expected = "Dictionary [size=10] <null, null, not, prepared, null, not, ?, you, null, not>";
            CHECK_EQ(formatted, expected);
        }
#endif
    }
}
