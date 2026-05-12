#include <oneapi/tbb/global_control.h>
#include <random>
#include <vector>

#include "../../src/array_id_func.h"
#include "../../src/chain.h"
#include "../../src/filter.h"
#include "../../src/flow_cutter_config.h"
#include "../../src/geo_pos.h"
#include "../../src/id_multi_func.h"
#include "../../src/min_fill_in.h"
#include "../../src/multi_arc.h"
#include "../../src/permutation.h"
#include "../../src/preorder.h"
#include "../../src/separator.h"

using namespace std;

template <class FuncA, class FuncB>
void run_inertial_flow_cutter(int const node_count,
                              std::vector<unsigned> const& v_tail,
                              std::vector<unsigned> const& v_head,
                              FuncA get_lonlat,
                              FuncB apply_level) {

  ArrayIDIDFunc tail, head;
  ArrayIDFunc<int> node_weight, arc_weight;

  ArrayIDIDFunc node_color, arc_color;
  ArrayIDFunc<GeoPos> node_geo_pos;

  ArrayIDIDFunc node_original_position;
  ArrayIDIDFunc arc_original_position;

  flow_cutter::Config flow_cutter_config;
  flow_cutter_config.thread_count = 4;

  auto arc_count = static_cast<int>(v_tail.size());

  auto const keep_arcs_if = [&](BitIDFunc const& keep_flag) {
    int new_arc_count = count_true(keep_flag);
    tail = keep_if(keep_flag, new_arc_count, move(tail));
    head = keep_if(keep_flag, new_arc_count, move(head));
    arc_weight = keep_if(keep_flag, new_arc_count, move(arc_weight));
    arc_color = keep_if(keep_flag, new_arc_count, move(arc_color));
    arc_original_position =
        keep_if(keep_flag, new_arc_count, std::move(arc_original_position));
  };

  auto const permutate_nodes = [&](ArrayIDIDFunc const& p) {
    auto inv_p = inverse_permutation(p);
    head = chain(std::move(head), inv_p);
    tail = chain(std::move(tail), inv_p);

    node_color = chain(p, std::move(node_color));
    node_geo_pos = chain(p, std::move(node_geo_pos));
    node_weight = chain(p, std::move(node_weight));
    node_original_position = chain(p, std::move(node_original_position));
  };

  auto const permutate_arcs = [&](ArrayIDIDFunc const& p) {
    tail = chain(p, move(tail));
    head = chain(p, move(head));
    arc_weight = chain(p, move(arc_weight));
    arc_color = chain(p, move(arc_color));
    arc_original_position = chain(p, std::move(arc_original_position));
  };

  auto const add_back_arcs = [&] {
    auto extended_tail =
        id_id_func(2 * tail.preimage_count(), tail.image_count(), [&](int i) {
          if (i < tail.preimage_count())
            return tail(i);
          else
            return head(i - tail.preimage_count());
        });
    auto extended_head =
        id_id_func(2 * tail.preimage_count(), tail.image_count(), [&](int i) {
          if (i < tail.preimage_count())
            return head(i);
          else
            return tail(i - tail.preimage_count());
        });

    auto extended_arc_weight = id_func(2 * tail.preimage_count(), [&](int i) {
      if (i < tail.preimage_count())
        return arc_weight(i);
      else
        return arc_weight(i - tail.preimage_count());
    });

    auto extended_arc_color = id_id_func(
        2 * tail.preimage_count(), arc_color.image_count(), [&](int i) {
          if (i < tail.preimage_count())
            return arc_color(i);
          else
            return arc_color(i - tail.preimage_count());
        });

    auto extended_arc_original_position =
        id_id_func(2 * tail.preimage_count(),
                   arc_original_position.image_count(), [&](int i) {
                     if (i < tail.preimage_count())
                       return arc_original_position(i);
                     else
                       return arc_original_position(i - tail.preimage_count());
                   });

    auto keep_flag = identify_non_multi_arcs(extended_tail, extended_head);

    for (int i = 0; i < tail.preimage_count(); ++i) keep_flag.set(i, true);

    int new_arc_count = count_true(keep_flag);
    ArrayIDIDFunc new_tail = keep_if(keep_flag, new_arc_count, extended_tail);
    ArrayIDIDFunc new_head = keep_if(keep_flag, new_arc_count, extended_head);
    ArrayIDFunc<int> new_arc_weight =
        keep_if(keep_flag, new_arc_count, extended_arc_weight);
    ArrayIDIDFunc new_arc_color =
        keep_if(keep_flag, new_arc_count, extended_arc_color);
    ArrayIDIDFunc new_original_position =
        keep_if(keep_flag, new_arc_count,
                extended_arc_original_position);  // this is weird...
    tail = move(new_tail);
    head = move(new_head);
    arc_weight = move(new_arc_weight);
    arc_color = move(new_arc_color);
    arc_original_position = move(new_original_position);
  };

  auto const remove_multi_arcs = [&] {
    keep_arcs_if(identify_non_multi_arcs(tail, head));
  };

  auto const remove_loops = [&] {
    keep_arcs_if(id_func(tail.preimage_count(),
                         [&](int i) { return head(i) != tail(i); }));
  };

  auto const reorder_nodes_at_random = [&] {
    ArrayIDIDFunc perm(tail.image_count(), tail.image_count());
    std::mt19937 rng(flow_cutter_config.random_seed);
    for (int i = 0; i < tail.image_count(); ++i) perm[i] = i;
    std::shuffle(perm.begin(), perm.end(), rng);
    permutate_nodes(perm);
  };
  auto const reorder_nodes_in_preorder = [&] {
    permutate_nodes(
        compute_preorder(compute_successor_function(tail, head)).first);
  };

  auto const sort_arcs = [&] {
    permutate_arcs(sort_arcs_first_by_tail_second_by_head(tail, head));
  };

  auto const reorder_nodes_in_accelerated_flow_cutter_cch_order = [&] {
    if (!is_symmetric(tail, head))
      throw runtime_error("Graph must be symmetric");
    if (has_multi_arcs(tail, head))
      throw runtime_error("Graph must not have multi arcs");
    if (!is_loop_free(tail, head))
      throw runtime_error("Graph must not have loops");

    ArrayIDIDFunc order;

    // omp_set_nested(true);
    // #pragma omp parallel num_threads(flow_cutter_config.thread_count)
    // #pragma omp single nowait
    {
      tbb::global_control gc(tbb::global_control::max_allowed_parallelism,
                             flow_cutter_config.thread_count);
      order = cch_order::compute_cch_graph_order(
          tail, head, arc_weight,
          flow_cutter::ComputeSeparator<flow_cutter_accelerated::CutterFactory,
                                        ArrayIDFunc<GeoPos>>(
              node_geo_pos, flow_cutter_config));
    }
    permutate_nodes(order);
  };

  tail = id_id_func(arc_count, node_count,
                    [&](unsigned i) { return static_cast<int>(v_tail[i]); });
  head = id_id_func(arc_count, node_count,
                    [&](unsigned i) { return static_cast<int>(v_head[i]); });
  node_weight = ArrayIDFunc<int>(node_count);
  node_weight.fill(0);
  arc_weight = ArrayIDFunc<int>(arc_count);
  arc_weight.fill(0);
  arc_original_position = identity_permutation(tail.preimage_count());

  node_color = ArrayIDIDFunc(tail.image_count(), 1);
  node_color.fill(0);
  node_geo_pos = ArrayIDFunc<GeoPos>(tail.image_count());
  for (int i = 0; i < node_count; ++i) {
    auto const lonlat = get_lonlat(i);
    node_geo_pos[i].lon = lonlat.first;
    node_geo_pos[i].lat = lonlat.second;
  }

  node_original_position = identity_permutation(tail.image_count());
  arc_color = ArrayIDIDFunc(tail.preimage_count(), 1);
  arc_color.fill(0);

  add_back_arcs();
  remove_multi_arcs();
  remove_loops();
  reorder_nodes_at_random();
  reorder_nodes_in_preorder();
  sort_arcs();
  reorder_nodes_in_accelerated_flow_cutter_cch_order();
  for (int i = 0; i < node_count; ++i) {
    apply_level(node_original_position[i], i);
  }
}