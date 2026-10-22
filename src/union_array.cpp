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

#include "sparrow/union_array.hpp"

#include <algorithm>
#include <iterator>
#include <limits>
#include <string>
#include <vector>

#include "sparrow/array.hpp"
#include "sparrow/debug/copy_tracker.hpp"
#include "sparrow/null_array.hpp"

namespace sparrow
{
    namespace
    {
        using dynamic_value = array_traits::value_type;
        using rebuild_value = detail::union_rebuild_value;
        using union_rebuild_payload = detail::union_rebuild_payload;

        /// @brief Finds an existing type ID not yet assigned, or TYPE_ID_MAP_SIZE if none is free.
        std::size_t next_free_type_id(std::span<const bool> used_type_ids)
        {
            return std::ranges::find(used_type_ids, false) - used_type_ids.begin();
        }

        /// @brief Resolves the child a plan entry targets: inserted values carry no child index,
        ///        only the index of the value they insert.
        std::size_t child_of(const rebuild_value& value, std::span<const std::size_t> inserted_children)
        {
            if (value.inserted)
            {
                SPARROW_ASSERT_TRUE(value.row < inserted_children.size());
                return inserted_children[value.row];
            }
            return value.child;
        }

        array make_empty_child(const array_wrapper& child)
        {
            const array view{child.get_arrow_proxy().view()};
            return array_empty_like(view);
        }

        /**
         * @brief Builds a null dynamic value holding the type of the given inner value.
         * @tparam T The type of the inner value (nullable alternatives are unwrapped by
         *           the caller, e.g. via get()).
         * @param value An inner value only used for its type.
         * @return A dynamic_value of the same type, in null state.
         */
        template <class T>
        dynamic_value make_null_dynamic_value(const T& value)
        {
            using stored_type = std::remove_cvref_t<T>;
            return dynamic_value(nullable<stored_type>(stored_type(value), false));
        }

        /**
         * @brief Creates a null-like dynamic value for the given dynamic value.
         * @param value The dynamic value for which to create a null-like value.
         * @return A dynamic_value representing the null-like value.
         */
        dynamic_value make_null_like(const dynamic_value& value)
        {
            return std::visit(
                [](const auto& typed_value) -> dynamic_value
                {
                    return make_null_dynamic_value(typed_value.get());
                },
#if SPARROW_GCC_11_2_WORKAROUND
                static_cast<const dynamic_value::base_type&>(value)
#else
                value
#endif
            );
        }

        /**
         * @brief Creates a rebuild payload for a dense union array.
         * @param values The rows to be rebuilt.
         * @param inserted_children Child index of each inserted value, in insertion order.
         * @param child_type_ids The type IDs of the child arrays.
         * @return A union_rebuild_payload containing the rebuilt data.
         */
        union_rebuild_payload make_dense_rebuild_payload(
            std::span<const rebuild_value> values,
            std::span<const std::size_t> inserted_children,
            std::span<const std::uint8_t> child_type_ids
        )
        {
            union_rebuild_payload payload;
            payload.child_values.resize(child_type_ids.size());
            payload.type_ids.reserve(values.size());
            payload.offsets.reserve(values.size());

            std::vector<std::size_t> child_value_counts(child_type_ids.size(), 0);
            for (const auto& value : values)
            {
                ++child_value_counts[child_of(value, inserted_children)];
            }
            for (std::size_t child_index = 0; child_index < child_type_ids.size(); ++child_index)
            {
                payload.child_values[child_index].reserve(child_value_counts[child_index]);
            }

            for (const auto& value : values)
            {
                const auto child_index = child_of(value, inserted_children);
                payload.type_ids.push_back(child_type_ids[child_index]);
                SPARROW_ASSERT_TRUE(
                    payload.child_values[child_index].size()
                    <= static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())
                );
                payload.offsets.push_back(static_cast<std::uint32_t>(payload.child_values[child_index].size()));
                payload.child_values[child_index].push_back(value);
            }
            return payload;
        }

        /**
         * @brief Creates a rebuild payload for a sparse union array.
         * @param values The rows to be rebuilt.
         * @param inserted_children Child index of each inserted value, in insertion order.
         * @param child_type_ids The type IDs of the child arrays.
         * @param fillers The row padding the non-selected output rows of each child.
         * @return A union_rebuild_payload containing the rebuilt data.
         */
        union_rebuild_payload make_sparse_rebuild_payload(
            std::span<const rebuild_value> values,
            std::span<const std::size_t> inserted_children,
            std::span<const std::uint8_t> child_type_ids,
            std::span<const rebuild_value> fillers
        )
        {
            union_rebuild_payload payload;
            payload.child_values.resize(child_type_ids.size());
            payload.type_ids.reserve(values.size());
            for (const auto& value : values)
            {
                payload.type_ids.push_back(child_type_ids[child_of(value, inserted_children)]);
            }

            for (std::size_t child_index = 0; child_index < child_type_ids.size(); ++child_index)
            {
                SPARROW_ASSERT_TRUE(child_index < fillers.size());
                auto& child_values = payload.child_values[child_index];
                child_values.reserve(values.size());
                for (const auto& value : values)
                {
                    child_values.push_back(
                        child_of(value, inserted_children) == child_index ? value : fillers[child_index]
                    );
                }
            }
            return payload;
        }
    }

    namespace copy_tracker
    {
        template <>
        SPARROW_API std::string key<dense_union_array>()
        {
            return "dense_union_array";
        }

        template <>
        SPARROW_API std::string key<sparse_union_array>()
        {
            return "sparse_union_array";
        }
    }

    /************************************
     * dense_union_array implementation *
     ************************************/

#ifdef __GNUC__
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wcast-align"
#endif

    dense_union_array::dense_union_array(arrow_proxy proxy)
        : base_type(std::move(proxy))
        , p_offsets(reinterpret_cast<std::int32_t*>(m_proxy.buffers()[1 /*index of offsets*/].data()))
    {
    }

    dense_union_array::dense_union_array(const dense_union_array& rhs)
        : dense_union_array(rhs.m_proxy)
    {
        copy_tracker::increase(copy_tracker::key<dense_union_array>());
    }

    dense_union_array& dense_union_array::operator=(const dense_union_array& rhs)
    {
        copy_tracker::increase(copy_tracker::key<dense_union_array>());
        if (this != &rhs)
        {
            base_type::operator=(rhs);
            p_offsets = reinterpret_cast<std::int32_t*>(m_proxy.buffers()[1 /*index of offsets*/].data());
        }
        return *this;
    }

#ifdef __GNUC__
#    pragma GCC diagnostic pop
#endif

    std::size_t dense_union_array::element_offset(std::size_t i) const
    {
        return static_cast<std::size_t>(p_offsets[i + m_proxy.offset()]);
    }

    auto dense_union_array::rebuild_in_place(
        std::vector<detail::union_rebuild_value> values,
        std::vector<array_traits::value_type> inserted,
        size_type return_index
    ) -> iterator
    {
        return union_array_crtp_base<dense_union_array>::rebuild_values(
            std::move(values),
            std::move(inserted),
            return_index
        );
    }

    /*************************************
     * sparse_union_array implementation *
     *************************************/

    sparse_union_array::sparse_union_array(arrow_proxy proxy)
        : base_type(std::move(proxy))
    {
    }

    sparse_union_array::sparse_union_array(const sparse_union_array& rhs)
        : base_type(rhs)
    {
        copy_tracker::increase(copy_tracker::key<sparse_union_array>());
    }

    sparse_union_array& sparse_union_array::operator=(const sparse_union_array& rhs)
    {
        copy_tracker::increase(copy_tracker::key<sparse_union_array>());
        if (this != &rhs)
        {
            base_type::operator=(rhs);
        }
        return *this;
    }

    std::size_t sparse_union_array::element_offset(std::size_t i) const
    {
        return i + m_proxy.offset();
    }

    auto sparse_union_array::rebuild_in_place(
        std::vector<detail::union_rebuild_value> values,
        std::vector<array_traits::value_type> inserted,
        size_type return_index
    ) -> iterator
    {
        return union_array_crtp_base<sparse_union_array>::rebuild_values(
            std::move(values),
            std::move(inserted),
            return_index
        );
    }

    /**
     * @brief Slices the row a filler rebuild value refers to, marked as null.
     *
     * Bitmap-less children (e.g. a nested union) cannot hold a null in a single row: the row
     * keeps its value, sparse-union filler rows being unspecified by the Arrow format.
     */
    array make_filler_row(const array& old_child_view, std::size_t row)
    {
        array filler = old_child_view.slice(row, row + 1);
        auto& proxy = detail::array_access::get_arrow_proxy(filler);
        if (auto& bitmap = proxy.bitmap(); bitmap.has_value())
        {
            *bitmap->begin() = false;
            proxy.set_null_count(1);
        }
        return filler;
    }

    /**
     * @brief Appends @p count consecutive rows of an old child, in a single insertion.
     */
    void append_old_child_values(array& destination, const array& old_child_view, std::size_t first_row, std::size_t count)
    {
        const array rows = old_child_view.slice_view(first_row, first_row + count);
        destination.insert(destination.cend(), rows.cbegin(), rows.cend());
    }

    /**
     * @brief Appends the rebuild rows of one child to @p destination.
     *
     * Rows reading consecutive rows of the old child are appended with a single range insertion,
     * and repeated rows (an inserted value inserted several times, or a filler row) with a single
     * repeated insertion..
     */
    template <typename CHILDREN>
    void append_rebuild_values(
        array& destination,
        std::span<const rebuild_value> values,
        std::span<const dynamic_value> inserted,
        std::span<const std::size_t> inserted_children,
        const CHILDREN& old_children
    )
    {
        // View over the old child the rows are read from, and the filler row reused by every
        // filler run of that child (interleaved filler rows would otherwise be re-sliced once
        // per row).
        std::optional<array> old_child_view;
        std::optional<array> filler_row;

        for (std::size_t index = 0; index < values.size();)
        {
            const auto child_index = child_of(values[index], inserted_children);
            const bool is_inserted = values[index].inserted;
            const bool force_null = values[index].force_null;
            const auto row = values[index].row;

            // A run repeats the same row (filler, inserted value) or reads consecutive rows.
            std::size_t count = 1;
            while (index + count < values.size())
            {
                const auto& next = values[index + count];
                if (child_of(next, inserted_children) != child_index || next.inserted != is_inserted
                    || next.force_null != force_null
                    || next.row != (force_null || is_inserted ? row : row + count))
                {
                    break;
                }
                ++count;
            }

            if (is_inserted)
            {
                const array inserted_row = array_make_from_element(inserted[row]);
                destination.insert(destination.cend(), inserted_row.cbegin(), inserted_row.cend(), count);
                index += count;
                continue;
            }

            if (!old_child_view.has_value())
            {
                old_child_view = array{old_children[child_index]->get_arrow_proxy().view()};
            }

            if (force_null)
            {
                if (!filler_row.has_value())
                {
                    filler_row = make_filler_row(*old_child_view, row);
                }
                destination.insert(destination.cend(), filler_row->cbegin(), filler_row->cend(), count);
            }
            else
            {
                append_old_child_values(destination, *old_child_view, row, count);
            }
            index += count;
        }
    }

    template <class DERIVED>
    auto union_array_crtp_base<DERIVED>::resolve_inserted_children(
        std::span<const array_traits::value_type> inserted,
        std::vector<union_child_schema>& child_schemas,
        std::vector<std::uint8_t>& child_type_ids,
        std::array<bool, TYPE_ID_MAP_SIZE>& used_type_ids,
        std::vector<array>& new_child_templates
    ) -> std::vector<std::size_t>
    {
        std::vector<std::size_t> inserted_children;
        inserted_children.reserve(inserted.size());
        for (const auto& value : inserted)
        {
            auto child = array_make_from_element(value);
            const union_child_schema schema{
                child.data_type(),
                std::string(detail::array_access::get_arrow_proxy(child).format())
            };
            const auto existing = std::ranges::find(child_schemas, schema);
            if (existing != child_schemas.end())
            {
                inserted_children.push_back(static_cast<std::size_t>(existing - child_schemas.begin()));
                continue;
            }

            SPARROW_ASSERT_TRUE(child_schemas.size() < TYPE_ID_MAP_SIZE);
            const auto type_id = next_free_type_id(used_type_ids);
            SPARROW_ASSERT_TRUE(type_id < TYPE_ID_MAP_SIZE);
            used_type_ids[type_id] = true;

            inserted_children.push_back(child_schemas.size());
            child_schemas.push_back(schema);
            child_type_ids.push_back(static_cast<std::uint8_t>(type_id));
            new_child_templates.push_back(std::move(child));
        }
        return inserted_children;
    }

    template <class DERIVED>
    auto union_array_crtp_base<DERIVED>::make_union_fillers(
        size_type old_child_count,
        const std::vector<union_child_schema>& child_schemas,
        std::vector<std::size_t>& inserted_children,
        std::vector<array_traits::value_type>& inserted
    ) const -> std::vector<rebuild_value>
    {
        std::vector<rebuild_value> fillers(child_schemas.size());
        for (std::size_t child_index = 0; child_index < fillers.size(); ++child_index)
        {
            if (child_index < old_child_count && array_size(*m_children[child_index]) > 0)
            {
                fillers[child_index] = rebuild_value{.child = child_index, .row = 0, .force_null = true};
                continue;
            }

            // A child created for an inserted value has no rows yet: derive its null from the
            // value that created it, an empty old child from its own default element.
            dynamic_value default_value;
            if (child_index < old_child_count)
            {
                default_value = array_default_value(*m_children[child_index]);
            }
            else
            {
                const auto creator = std::ranges::find(inserted_children, child_index);
                SPARROW_ASSERT_TRUE(creator != inserted_children.end());
                default_value = inserted[static_cast<std::size_t>(creator - inserted_children.begin())];
            }
            inserted.push_back(make_null_like(default_value));
            // Keep the two parallel lists in sync: entries refer to the filler by inserted value
            // index, and their child index is read back from inserted_children.
            inserted_children.push_back(child_index);
            fillers[child_index] = rebuild_value{.row = inserted.size() - 1, .inserted = true};
        }
        return fillers;
    }

    template <class DERIVED>
    auto union_array_crtp_base<DERIVED>::make_rebuild_payload(
        std::span<const rebuild_value> values,
        std::span<const std::size_t> inserted_children,
        std::span<const std::uint8_t> child_type_ids,
        std::span<const rebuild_value> fillers
    ) const -> union_rebuild_payload
    {
        if constexpr (is_dense_union_array_v<DERIVED>)
        {
            return make_dense_rebuild_payload(values, inserted_children, child_type_ids);
        }
        else
        {
            return make_sparse_rebuild_payload(values, inserted_children, child_type_ids, fillers);
        }
    }

    template <class DERIVED>
    auto union_array_crtp_base<DERIVED>::build_rebuilt_children(
        const detail::union_rebuild_payload& payload,
        std::span<const array_traits::value_type> inserted,
        std::span<const std::size_t> inserted_children,
        const children_type& old_children,
        std::vector<array>& new_child_templates
    ) const -> std::vector<array>
    {
        const auto old_child_count = old_children.size();
        std::vector<array> new_children;
        new_children.reserve(payload.child_values.size());
        for (std::size_t child_index = 0; child_index < payload.child_values.size(); ++child_index)
        {
            array child = child_index < old_child_count
                              ? make_empty_child(*old_children[child_index])
                              : std::move(new_child_templates[child_index - old_child_count]);
            auto& values_for_child = payload.child_values[child_index];
            if (child.data_type() == data_type::NA)
            {
                const auto& child_proxy = detail::array_access::get_arrow_proxy(child);
                child = array(null_array(
                    values_for_child.size(),
                    child_proxy.name(),
                    child_proxy.metadata()
                ));
            }
            else
            {
                child.erase(child.cbegin(), child.cend());
                append_rebuild_values(child, values_for_child, inserted, inserted_children, old_children);
            }
            new_children.push_back(std::move(child));
        }
        return new_children;
    }

    template <class DERIVED>
    auto union_array_crtp_base<DERIVED>::create_rebuild_proxy(
        std::vector<array> children,
        detail::union_rebuild_payload&& payload,
        std::span<const std::uint8_t> child_type_ids
    ) const -> arrow_proxy
    {
        auto metadata = m_proxy.metadata();
        std::optional<std::vector<std::uint8_t>> type_id_mapping{
            std::vector<std::uint8_t>{child_type_ids.begin(), child_type_ids.end()}
        };
        if constexpr (is_dense_union_array_v<DERIVED>)
        {
            return DERIVED::create_proxy(
                std::move(children),
                std::move(payload.type_ids),
                std::move(payload.offsets),
                std::move(type_id_mapping),
                m_proxy.name(),
                std::move(metadata)
            );
        }
        else
        {
            return DERIVED::create_proxy(
                std::move(children),
                std::move(payload.type_ids),
                std::move(type_id_mapping),
                m_proxy.name(),
                std::move(metadata)
            );
        }
    }

    template <class DERIVED>
    auto union_array_crtp_base<DERIVED>::adopt_rebuilt_array(
        arrow_proxy replacement,
        std::vector<std::uint8_t> child_type_ids,
        size_type return_index
    ) -> iterator
    {
        m_proxy = std::move(replacement);
        p_type_ids = reinterpret_cast<std::uint8_t*>(m_proxy.buffers()[0].data());
        m_children = make_children(m_proxy);
        m_child_type_ids = std::move(child_type_ids);
        m_type_id_map = make_type_id_map(m_child_type_ids);
        if constexpr (is_dense_union_array_v<DERIVED>)
        {
            this->derived_cast().p_offsets = reinterpret_cast<std::int32_t*>(m_proxy.buffers()[1].data());
        }
        return iterator(functor_type{&this->derived_cast()}, return_index);
    }

    template <class DERIVED>
    auto union_array_crtp_base<DERIVED>::rebuild_values(
        std::vector<rebuild_value> values,
        std::vector<dynamic_value> inserted,
        size_type return_index
    ) -> iterator
    {
        const auto old_child_count = m_children.size();

        std::vector<std::uint8_t> child_type_ids;
        std::array<bool, TYPE_ID_MAP_SIZE> used_type_ids{};
        std::vector<union_child_schema> child_schemas = make_child_schemas(child_type_ids, used_type_ids);

        // Inserted values are resolved once each, however often the plan refers to them.
        std::vector<array> new_child_templates;
        std::vector<std::size_t> inserted_children = resolve_inserted_children(
            inserted,
            child_schemas,
            child_type_ids,
            used_type_ids,
            new_child_templates
        );

        std::vector<rebuild_value> fillers;
        if constexpr (!is_dense_union_array_v<DERIVED>)
        {
            fillers = make_union_fillers(old_child_count, child_schemas, inserted_children, inserted);
        }

        union_rebuild_payload payload = make_rebuild_payload(
            std::span<const rebuild_value>{values},
            std::span<const std::size_t>{inserted_children},
            child_type_ids,
            std::span<const rebuild_value>{fillers}
        );

        std::vector<array> new_children = build_rebuilt_children(
            payload,
            inserted,
            inserted_children,
            m_children,
            new_child_templates
        );

        auto replacement = create_rebuild_proxy(std::move(new_children), std::move(payload), child_type_ids);
        return adopt_rebuilt_array(std::move(replacement), std::move(child_type_ids), return_index);
    }

    template <class DERIVED>
    auto union_array_crtp_base<DERIVED>::insert(const_iterator pos, const_reference value, size_type count)
        -> iterator
    {
        // A no-op insertion must not require offset zero (see insert_materialized).
        if (count == 0)
        {
            return iterator_at(static_cast<size_type>(pos - cbegin()));
        }
        return insert(pos, array_traits::value_type(value), count);
    }

    template <class DERIVED>
    auto union_array_crtp_base<DERIVED>::insert(
        const_iterator pos,
        const array_traits::value_type& value,
        size_type count
    ) -> iterator
    {
        // A no-op insertion must not require offset zero (see insert_materialized).
        if (count == 0)
        {
            return iterator_at(static_cast<size_type>(pos - cbegin()));
        }
        return insert_materialized(pos, std::vector<array_traits::value_type>{value}, count);
    }

    template <class DERIVED>
    auto union_array_crtp_base<DERIVED>::insert_materialized(
        const_iterator pos,
        std::vector<array_traits::value_type> inserted,
        size_type count
    ) -> iterator
    {
        const auto pos_index = static_cast<size_type>(pos - cbegin());
        if (inserted.empty())
        {
            return iterator_at(pos_index);
        }

        SPARROW_ASSERT_TRUE(count > 0);
        SPARROW_ASSERT_TRUE(m_proxy.offset() == 0);
        SPARROW_ASSERT_TRUE(pos_index <= size());

        const auto old_size = size();
        const auto value_count = static_cast<size_type>(inserted.size());
        std::vector<detail::union_rebuild_value> entries;
        entries.reserve(old_size + value_count * count);

        append_rebuild_entries(entries, 0, pos_index);
        for (size_type repetition = 0; repetition < count; ++repetition)
        {
            for (size_type value_index = 0; value_index < value_count; ++value_index)
            {
                entries.push_back(detail::union_rebuild_value{
                    .row = value_index,
                    .inserted = true
                });
            }
        }
        append_rebuild_entries(entries, pos_index, old_size);
        return this->derived_cast().rebuild_in_place(std::move(entries), std::move(inserted), pos_index);
    }

    template SPARROW_API auto union_array_crtp_base<dense_union_array>::insert(
        const_iterator pos,
        const_reference value,
        size_type count
    ) -> iterator;
    template SPARROW_API auto union_array_crtp_base<dense_union_array>::insert(
        const_iterator pos,
        const array_traits::value_type& value,
        size_type count
    ) -> iterator;
    template SPARROW_API auto union_array_crtp_base<dense_union_array>::insert_materialized(
        const_iterator pos,
        std::vector<array_traits::value_type> inserted,
        size_type count
    ) -> iterator;
    template SPARROW_API auto union_array_crtp_base<sparse_union_array>::insert(
        const_iterator pos,
        const_reference value,
        size_type count
    ) -> iterator;
    template SPARROW_API auto union_array_crtp_base<sparse_union_array>::insert(
        const_iterator pos,
        const array_traits::value_type& value,
        size_type count
    ) -> iterator;
    template SPARROW_API auto union_array_crtp_base<sparse_union_array>::insert_materialized(
        const_iterator pos,
        std::vector<array_traits::value_type> inserted,
        size_type count
    ) -> iterator;

    template SPARROW_API auto union_array_crtp_base<dense_union_array>::make_union_fillers(
        size_type old_child_count,
        const std::vector<union_child_schema>& child_schemas,
        std::vector<std::size_t>& inserted_children,
        std::vector<array_traits::value_type>& inserted
    ) const -> std::vector<detail::union_rebuild_value>;
    template SPARROW_API auto union_array_crtp_base<dense_union_array>::make_rebuild_payload(
        std::span<const detail::union_rebuild_value> values,
        std::span<const std::size_t> inserted_children,
        std::span<const std::uint8_t> child_type_ids,
        std::span<const detail::union_rebuild_value> fillers
    ) const -> detail::union_rebuild_payload;
    template SPARROW_API auto union_array_crtp_base<dense_union_array>::create_rebuild_proxy(
        std::vector<array> children,
        detail::union_rebuild_payload&& payload,
        std::span<const std::uint8_t> child_type_ids
    ) const -> arrow_proxy;
    template SPARROW_API auto union_array_crtp_base<dense_union_array>::build_rebuilt_children(
        const detail::union_rebuild_payload& payload,
        std::span<const array_traits::value_type> inserted,
        std::span<const std::size_t> inserted_children,
        const union_array_crtp_base<dense_union_array>::children_type& old_children,
        std::vector<array>& new_child_templates
    ) const -> std::vector<array>;
    template SPARROW_API auto union_array_crtp_base<dense_union_array>::adopt_rebuilt_array(
        arrow_proxy replacement,
        std::vector<std::uint8_t> child_type_ids,
        size_type return_index
    ) -> iterator;
    template SPARROW_API auto union_array_crtp_base<dense_union_array>::rebuild_values(
        std::vector<detail::union_rebuild_value> values,
        std::vector<array_traits::value_type> inserted,
        size_type return_index
    ) -> iterator;
    template SPARROW_API auto union_array_crtp_base<sparse_union_array>::make_union_fillers(
        size_type old_child_count,
        const std::vector<union_child_schema>& child_schemas,
        std::vector<std::size_t>& inserted_children,
        std::vector<array_traits::value_type>& inserted
    ) const -> std::vector<detail::union_rebuild_value>;
    template SPARROW_API auto union_array_crtp_base<sparse_union_array>::make_rebuild_payload(
        std::span<const detail::union_rebuild_value> values,
        std::span<const std::size_t> inserted_children,
        std::span<const std::uint8_t> child_type_ids,
        std::span<const detail::union_rebuild_value> fillers
    ) const -> detail::union_rebuild_payload;
    template SPARROW_API auto union_array_crtp_base<sparse_union_array>::create_rebuild_proxy(
        std::vector<array> children,
        detail::union_rebuild_payload&& payload,
        std::span<const std::uint8_t> child_type_ids
    ) const -> arrow_proxy;
    template SPARROW_API auto union_array_crtp_base<sparse_union_array>::build_rebuilt_children(
        const detail::union_rebuild_payload& payload,
        std::span<const array_traits::value_type> inserted,
        std::span<const std::size_t> inserted_children,
        const union_array_crtp_base<sparse_union_array>::children_type& old_children,
        std::vector<array>& new_child_templates
    ) const -> std::vector<array>;
    template SPARROW_API auto union_array_crtp_base<sparse_union_array>::adopt_rebuilt_array(
        arrow_proxy replacement,
        std::vector<std::uint8_t> child_type_ids,
        size_type return_index
    ) -> iterator;
    template SPARROW_API auto union_array_crtp_base<sparse_union_array>::rebuild_values(
        std::vector<detail::union_rebuild_value> values,
        std::vector<array_traits::value_type> inserted,
        size_type return_index
    ) -> iterator;
}
