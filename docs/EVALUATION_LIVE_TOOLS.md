# Live Tools Shadow Testing 报告

## 结论

- provider：`amap`
- mode：`replay`
- agent real multi-agent：False
- llm：`qwen` / `qwen3.7-plus`
- layered_enabled：False（Step 4 前的历史字段；新架构已删除该开关，固定 Multi-Agent Full = V3）
- provider configured：True
- snapshot root：`/Users/carrier/run/projects/travel-agent/data/eval/live_tools/snapshots/amap/shadow-full`
- snapshot version：`474ec63eaa23f260c0b097ca7d14c96036dd3013664c7e679096de0d4721419e`
- API success rate：1.0
- operation success rate：{'budget_estimate': 1.0, 'hotel_search': 1.0, 'poi_search': 1.0, 'restaurant_search': 1.0, 'route': 1.0, 'weather': 1.0}
- fallback rate：0.1233
- fallback reason breakdown：{'primary_unusable': 74}
- primary success rate：1.0
- primary usable rate：0.8767
- timeout rate：0.0
- empty result rate：0.1
- expected empty pass rate：1.0
- duplicate POI rate：0.0
- route unavailable rate：0.14
- unexpected route unavailable rate：0.0444
- route degraded rate：0.04
- schema valid rate：1.0
- avg category match rate：0.9657
- restaurant / hotel / budget success rate：1.0 / 1.0 / 1.0
- restaurant / hotel category match rate：0.973 / 1.0
- avg city contamination rate：0.0
- latency p50 / p95：256.8 / 817.71 ms

## 明细

| case | city | operation | expect_empty | ok | fallback | reason | primary_ok | primary_usable | empty | route_unavailable | route_source | latency_ms | count | dup | cat_match | city_bad | schema | primary_error | error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| 北京_food | 北京 | poi_search | False | True | False | None | True | True | False | False |  | 491.64 | 20 | 0 | 1.0000 | 0.0000 | True |  |  |
| 北京_food | 北京 | weather | False | True | False | None | True | True | False | False |  | 225.54 | None | 0 |  |  | True |  |  |
| 北京_food | 北京 | route | False | True | False | None | True | True | False | False | amap | 707.23 | None | 0 |  |  | True |  |  |
| 北京_food | 北京 | restaurant_search | False | True | False | None | True | True | False | False |  | 306.53 | 10 | 0 | 1.0000 | 0.0000 | True |  |  |
| 北京_food | 北京 | hotel_search | False | True | False | None | True | True | False | False |  | 303.47 | 10 | 0 | 1.0000 | 0.0000 | True |  |  |
| 北京_food | 北京 | budget_estimate | False | True | False | None | True | True | False | False |  | 0.51 | 1 | 0 |  |  | True |  |  |
| 北京_history | 北京 | poi_search | False | True | False | None | True | True | False | False |  | 175.78 | 2 | 0 | 1.0000 | 0.0000 | True |  |  |
| 北京_history | 北京 | weather | False | True | False | None | True | True | False | False |  | 218.34 | None | 0 |  |  | True |  |  |
| 北京_history | 北京 | route | False | True | False | None | True | True | False | False | amap | 647.09 | None | 0 |  |  | True |  |  |
| 北京_history | 北京 | restaurant_search | False | True | False | None | True | True | False | False |  | 312.8 | 10 | 0 | 1.0000 | 0.0000 | True |  |  |
| 北京_history | 北京 | hotel_search | False | True | False | None | True | True | False | False |  | 305.34 | 10 | 0 | 1.0000 | 0.0000 | True |  |  |
| 北京_history | 北京 | budget_estimate | False | True | False | None | True | True | False | False |  | 0.45 | 1 | 0 |  |  | True |  |  |
| 北京_nature | 北京 | poi_search | False | True | False | None | True | True | False | False |  | 484.44 | 20 | 0 | 0.9500 | 0.0000 | True |  |  |
| 北京_nature | 北京 | weather | False | True | False | None | True | True | False | False |  | 370.5 | None | 0 |  |  | True |  |  |
| 北京_nature | 北京 | route | False | True | False | None | True | True | False | False | amap | 1490.42 | None | 0 |  |  | True |  |  |
| 北京_nature | 北京 | restaurant_search | False | True | False | None | True | True | False | False |  | 566.89 | 10 | 0 | 1.0000 | 0.0000 | True |  |  |
| 北京_nature | 北京 | hotel_search | False | True | False | None | True | True | False | False |  | 304.09 | 10 | 0 | 1.0000 | 0.0000 | True |  |  |
| 北京_nature | 北京 | budget_estimate | False | True | False | None | True | True | False | False |  | 0.99 | 1 | 0 |  |  | True |  |  |
| 北京_shopping | 北京 | poi_search | False | True | False | None | True | True | False | False |  | 308.8 | 20 | 0 | 0.9500 | 0.0000 | True |  |  |
| 北京_shopping | 北京 | weather | False | True | False | None | True | True | False | False |  | 224.95 | None | 0 |  |  | True |  |  |
| 北京_shopping | 北京 | route | False | True | False | None | True | True | False | False | amap | 648.95 | None | 0 |  |  | True |  |  |
| 北京_shopping | 北京 | restaurant_search | False | True | False | None | True | True | False | False |  | 296.52 | 10 | 0 | 1.0000 | 0.0000 | True |  |  |
| 北京_shopping | 北京 | hotel_search | False | True | False | None | True | True | False | False |  | 315.19 | 10 | 0 | 1.0000 | 0.0000 | True |  |  |
| 北京_shopping | 北京 | budget_estimate | False | True | False | None | True | True | False | False |  | 1.2 | 1 | 0 |  |  | True |  |  |
| 北京_family | 北京 | poi_search | False | True | False | None | True | True | False | False |  | 220.79 | 9 | 0 | 0.8889 | 0.0000 | True |  |  |
| 北京_family | 北京 | weather | False | True | False | None | True | True | False | False |  | 263.61 | None | 0 |  |  | True |  |  |
| 北京_family | 北京 | route | False | True | False | None | True | True | False | False | amap | 816.76 | None | 0 |  |  | True |  |  |
| 北京_family | 北京 | restaurant_search | False | True | False | None | True | True | False | False |  | 304.27 | 10 | 0 | 1.0000 | 0.0000 | True |  |  |
| 北京_family | 北京 | hotel_search | False | True | False | None | True | True | False | False |  | 276.26 | 10 | 0 | 1.0000 | 0.0000 | True |  |  |
| 北京_family | 北京 | budget_estimate | False | True | False | None | True | True | False | False |  | 0.77 | 1 | 0 |  |  | True |  |  |
| 北京_museum | 北京 | poi_search | False | True | False | None | True | True | False | False |  | 374.51 | 20 | 0 | 1.0000 | 0.0000 | True |  |  |
| 北京_museum | 北京 | weather | False | True | False | None | True | True | False | False |  | 226.73 | None | 0 |  |  | True |  |  |
| 北京_museum | 北京 | route | False | True | False | None | True | True | False | False | amap | 601.26 | None | 0 |  |  | True |  |  |
| 北京_museum | 北京 | restaurant_search | False | True | False | None | True | True | False | False |  | 306.11 | 10 | 0 | 1.0000 | 0.0000 | True |  |  |
| 北京_museum | 北京 | hotel_search | False | True | False | None | True | True | False | False |  | 271.39 | 10 | 0 | 1.0000 | 0.0000 | True |  |  |
| 北京_museum | 北京 | budget_estimate | False | True | False | None | True | True | False | False |  | 0.76 | 1 | 0 |  |  | True |  |  |
| 北京_night_food | 北京 | poi_search | False | True | False | None | True | True | False | False |  | 286.01 | 20 | 0 | 1.0000 | 0.0000 | True |  |  |
| 北京_night_food | 北京 | weather | False | True | False | None | True | True | False | False |  | 251.86 | None | 0 |  |  | True |  |  |
| 北京_night_food | 北京 | route | False | True | False | None | True | True | False | False | amap | 646.34 | None | 0 |  |  | True |  |  |
| 北京_night_food | 北京 | restaurant_search | False | True | False | None | True | True | False | False |  | 220.06 | 10 | 0 | 1.0000 | 0.0000 | True |  |  |
| 北京_night_food | 北京 | hotel_search | False | True | False | None | True | True | False | False |  | 267.61 | 10 | 0 | 1.0000 | 0.0000 | True |  |  |
| 北京_night_food | 北京 | budget_estimate | False | True | False | None | True | True | False | False |  | 1.3 | 1 | 0 |  |  | True |  |  |
| 北京_route_long | 北京 | poi_search | False | True | False | None | True | True | False | False |  | 436.21 | 10 | 0 | 1.0000 | 0.0000 | True |  |  |
| 北京_route_long | 北京 | weather | False | True | False | None | True | True | False | False |  | 243.15 | None | 0 |  |  | True |  |  |
| 北京_route_long | 北京 | route | False | True | False | None | True | True | False | False | amap | 965.44 | None | 0 |  |  | True |  |  |
| 北京_route_long | 北京 | restaurant_search | False | True | False | None | True | True | False | False |  | 311.53 | 10 | 0 | 1.0000 | 0.0000 | True |  |  |
| 北京_route_long | 北京 | hotel_search | False | True | False | None | True | True | False | False |  | 274.08 | 10 | 0 | 1.0000 | 0.0000 | True |  |  |
| 北京_route_long | 北京 | budget_estimate | False | True | False | None | True | True | False | False |  | 1.17 | 1 | 0 |  |  | True |  |  |
| 上海_food | 上海 | poi_search | False | True | False | None | True | True | False | False |  | 473.25 | 20 | 0 | 1.0000 | 0.0000 | True |  |  |
| 上海_food | 上海 | weather | False | True | False | None | True | True | False | False |  | 258.66 | None | 0 |  |  | True |  |  |
| 上海_food | 上海 | route | False | True | False | None | True | True | False | False | amap | 939.44 | None | 0 |  |  | True |  |  |
| 上海_food | 上海 | restaurant_search | False | True | False | None | True | True | False | False |  | 632.94 | 10 | 0 | 1.0000 | 0.0000 | True |  |  |
| 上海_food | 上海 | hotel_search | False | True | False | None | True | True | False | False |  | 334.42 | 10 | 0 | 1.0000 | 0.0000 | True |  |  |
| 上海_food | 上海 | budget_estimate | False | True | False | None | True | True | False | False |  | 0.63 | 1 | 0 |  |  | True |  |  |
| 上海_history | 上海 | poi_search | False | True | False | None | True | True | False | False |  | 186.41 | 3 | 0 | 1.0000 | 0.0000 | True |  |  |
| 上海_history | 上海 | weather | False | True | False | None | True | True | False | False |  | 250.23 | None | 0 |  |  | True |  |  |
| 上海_history | 上海 | route | False | True | False | None | True | True | False | False | amap | 889.47 | None | 0 |  |  | True |  |  |
| 上海_history | 上海 | restaurant_search | False | True | False | None | True | True | False | False |  | 282.82 | 5 | 0 | 1.0000 | 0.0000 | True |  |  |
| 上海_history | 上海 | hotel_search | False | True | False | None | True | True | False | False |  | 288.81 | 10 | 0 | 1.0000 | 0.0000 | True |  |  |
| 上海_history | 上海 | budget_estimate | False | True | False | None | True | True | False | False |  | 0.93 | 1 | 0 |  |  | True |  |  |
| 上海_nature | 上海 | poi_search | False | True | False | None | True | True | False | False |  | 426.45 | 20 | 0 | 1.0000 | 0.0000 | True |  |  |
| 上海_nature | 上海 | weather | False | True | False | None | True | True | False | False |  | 215.39 | None | 0 |  |  | True |  |  |
| 上海_nature | 上海 | route | False | True | False | None | True | True | False | False | amap | 933.98 | None | 0 |  |  | True |  |  |
| 上海_nature | 上海 | restaurant_search | False | True | False | None | True | True | False | False |  | 244.18 | 5 | 0 | 1.0000 | 0.0000 | True |  |  |
| 上海_nature | 上海 | hotel_search | False | True | False | None | True | True | False | False |  | 315.33 | 10 | 0 | 1.0000 | 0.0000 | True |  |  |
| 上海_nature | 上海 | budget_estimate | False | True | False | None | True | True | False | False |  | 0.61 | 1 | 0 |  |  | True |  |  |
| 上海_shopping | 上海 | poi_search | False | True | False | None | True | True | False | False |  | 323.56 | 20 | 0 | 0.9000 | 0.0000 | True |  |  |
| 上海_shopping | 上海 | weather | False | True | False | None | True | True | False | False |  | 220.29 | None | 0 |  |  | True |  |  |
| 上海_shopping | 上海 | route | False | True | False | None | True | True | False | False | amap | 583.89 | None | 0 |  |  | True |  |  |
| 上海_shopping | 上海 | restaurant_search | False | True | False | None | True | True | False | False |  | 248.82 | 5 | 0 | 1.0000 | 0.0000 | True |  |  |
| 上海_shopping | 上海 | hotel_search | False | True | False | None | True | True | False | False |  | 299.22 | 10 | 0 | 1.0000 | 0.0000 | True |  |  |
| 上海_shopping | 上海 | budget_estimate | False | True | False | None | True | True | False | False |  | 0.77 | 1 | 0 |  |  | True |  |  |
| 上海_family | 上海 | poi_search | False | True | False | None | True | True | False | False |  | 203.97 | 12 | 0 | 1.0000 | 0.0000 | True |  |  |
| 上海_family | 上海 | weather | False | True | False | None | True | True | False | False |  | 221.46 | None | 0 |  |  | True |  |  |
| 上海_family | 上海 | route | False | True | True | primary_unusable | True | False | True | True |  | 248.3 | None | 0 |  |  | True |  |  |
| 上海_family | 上海 | restaurant_search | False | True | False | None | True | True | False | False |  | 225.46 | 5 | 0 | 1.0000 | 0.0000 | True |  |  |
| 上海_family | 上海 | hotel_search | False | True | False | None | True | True | False | False |  | 329.41 | 10 | 0 | 1.0000 | 0.0000 | True |  |  |
| 上海_family | 上海 | budget_estimate | False | True | False | None | True | True | False | False |  | 0.39 | 1 | 0 |  |  | True |  |  |
| 上海_museum | 上海 | poi_search | False | True | False | None | True | True | False | False |  | 347.03 | 20 | 0 | 1.0000 | 0.0000 | True |  |  |
| 上海_museum | 上海 | weather | False | True | False | None | True | True | False | False |  | 252.45 | None | 0 |  |  | True |  |  |
| 上海_museum | 上海 | route | False | True | False | None | True | True | False | False | amap | 729.3 | None | 0 |  |  | True |  |  |
| 上海_museum | 上海 | restaurant_search | False | True | False | None | True | True | False | False |  | 213.41 | 5 | 0 | 1.0000 | 0.0000 | True |  |  |
| 上海_museum | 上海 | hotel_search | False | True | False | None | True | True | False | False |  | 270.44 | 10 | 0 | 1.0000 | 0.0000 | True |  |  |
| 上海_museum | 上海 | budget_estimate | False | True | False | None | True | True | False | False |  | 0.8 | 1 | 0 |  |  | True |  |  |
| 上海_night_food | 上海 | poi_search | False | True | False | None | True | True | False | False |  | 277.88 | 20 | 0 | 1.0000 | 0.0000 | True |  |  |
| 上海_night_food | 上海 | weather | False | True | False | None | True | True | False | False |  | 231.44 | None | 0 |  |  | True |  |  |
| 上海_night_food | 上海 | route | False | True | False | None | True | True | False | False | amap | 640.31 | None | 0 |  |  | True |  |  |
| 上海_night_food | 上海 | restaurant_search | False | True | False | None | True | True | False | False |  | 238.64 | 10 | 0 | 1.0000 | 0.0000 | True |  |  |
| 上海_night_food | 上海 | hotel_search | False | True | False | None | True | True | False | False |  | 317.97 | 10 | 0 | 1.0000 | 0.0000 | True |  |  |
| 上海_night_food | 上海 | budget_estimate | False | True | False | None | True | True | False | False |  | 0.55 | 1 | 0 |  |  | True |  |  |
| 上海_route_long | 上海 | poi_search | False | True | False | None | True | True | False | False |  | 301.39 | 9 | 0 | 1.0000 | 0.0000 | True |  |  |
| 上海_route_long | 上海 | weather | False | True | False | None | True | True | False | False |  | 482.62 | None | 0 |  |  | True |  |  |
| 上海_route_long | 上海 | route | False | True | False | None | True | True | False | False | amap | 1560.12 | None | 0 |  |  | True |  |  |
| 上海_route_long | 上海 | restaurant_search | False | True | False | None | True | True | False | False |  | 240.98 | 5 | 0 | 1.0000 | 0.0000 | True |  |  |
| 上海_route_long | 上海 | hotel_search | False | True | False | None | True | True | False | False |  | 322.7 | 10 | 0 | 1.0000 | 0.0000 | True |  |  |
| 上海_route_long | 上海 | budget_estimate | False | True | False | None | True | True | False | False |  | 0.51 | 1 | 0 |  |  | True |  |  |
| 杭州_food | 杭州 | poi_search | False | True | False | None | True | True | False | False |  | 464.63 | 20 | 0 | 0.9500 | 0.0000 | True |  |  |
| 杭州_food | 杭州 | weather | False | True | False | None | True | True | False | False |  | 209.71 | None | 0 |  |  | True |  |  |
| 杭州_food | 杭州 | route | False | True | False | None | True | True | False | False | amap | 712.73 | None | 0 |  |  | True |  |  |
| 杭州_food | 杭州 | restaurant_search | False | True | False | None | True | True | False | False |  | 253.55 | 10 | 0 | 1.0000 | 0.0000 | True |  |  |
| 杭州_food | 杭州 | hotel_search | False | True | False | None | True | True | False | False |  | 259.59 | 10 | 0 | 1.0000 | 0.0000 | True |  |  |
| 杭州_food | 杭州 | budget_estimate | False | True | False | None | True | True | False | False |  | 0.42 | 1 | 0 |  |  | True |  |  |
| 杭州_history | 杭州 | poi_search | False | True | True | primary_unusable | True | False | False | False |  | 97.35 | 4 | 0 | 1.0000 | 0.0000 | True |  |  |
| 杭州_history | 杭州 | weather | False | True | False | None | True | True | False | False |  | 248.71 | None | 0 |  |  | True |  |  |
| 杭州_history | 杭州 | route | False | True | False | None | True | True | False | False | amap | 793.7 | None | 0 |  |  | True |  |  |
| 杭州_history | 杭州 | restaurant_search | False | True | False | None | True | True | False | False |  | 227.43 | 4 | 0 | 1.0000 | 0.0000 | True |  |  |
| 杭州_history | 杭州 | hotel_search | False | True | False | None | True | True | False | False |  | 256.8 | 10 | 0 | 1.0000 | 0.0000 | True |  |  |
| 杭州_history | 杭州 | budget_estimate | False | True | False | None | True | True | False | False |  | 0.55 | 1 | 0 |  |  | True |  |  |
| 杭州_nature | 杭州 | poi_search | False | True | False | None | True | True | False | False |  | 438.12 | 20 | 0 | 1.0000 | 0.0000 | True |  |  |
| 杭州_nature | 杭州 | weather | False | True | False | None | True | True | False | False |  | 276.31 | None | 0 |  |  | True |  |  |
| 杭州_nature | 杭州 | route | False | True | False | None | True | True | False | False | amap_fallback_estimate | 817.53 | None | 0 |  |  | True |  |  |
| 杭州_nature | 杭州 | restaurant_search | False | True | False | None | True | True | False | False |  | 254.91 | 4 | 0 | 1.0000 | 0.0000 | True |  |  |
| 杭州_nature | 杭州 | hotel_search | False | True | False | None | True | True | False | False |  | 362.03 | 10 | 0 | 1.0000 | 0.0000 | True |  |  |
| 杭州_nature | 杭州 | budget_estimate | False | True | False | None | True | True | False | False |  | 0.92 | 1 | 0 |  |  | True |  |  |
| 杭州_shopping | 杭州 | poi_search | False | True | False | None | True | True | False | False |  | 300.29 | 20 | 0 | 0.8500 | 0.0000 | True |  |  |
| 杭州_shopping | 杭州 | weather | False | True | False | None | True | True | False | False |  | 236.37 | None | 0 |  |  | True |  |  |
| 杭州_shopping | 杭州 | route | False | True | False | None | True | True | False | False | amap | 587.61 | None | 0 |  |  | True |  |  |
| 杭州_shopping | 杭州 | restaurant_search | False | True | False | None | True | True | False | False |  | 218.83 | 4 | 0 | 1.0000 | 0.0000 | True |  |  |
| 杭州_shopping | 杭州 | hotel_search | False | True | False | None | True | True | False | False |  | 385.8 | 10 | 0 | 1.0000 | 0.0000 | True |  |  |
| 杭州_shopping | 杭州 | budget_estimate | False | True | False | None | True | True | False | False |  | 1.01 | 1 | 0 |  |  | True |  |  |
| 杭州_family | 杭州 | poi_search | False | True | False | None | True | True | False | False |  | 244.71 | 8 | 0 | 1.0000 | 0.0000 | True |  |  |
| 杭州_family | 杭州 | weather | False | True | False | None | True | True | False | False |  | 264.35 | None | 0 |  |  | True |  |  |
| 杭州_family | 杭州 | route | False | True | False | None | True | True | False | False | amap | 686.64 | None | 0 |  |  | True |  |  |
| 杭州_family | 杭州 | restaurant_search | False | True | False | None | True | True | False | False |  | 232.88 | 4 | 0 | 1.0000 | 0.0000 | True |  |  |
| 杭州_family | 杭州 | hotel_search | False | True | False | None | True | True | False | False |  | 261.6 | 10 | 0 | 1.0000 | 0.0000 | True |  |  |
| 杭州_family | 杭州 | budget_estimate | False | True | False | None | True | True | False | False |  | 1.39 | 1 | 0 |  |  | True |  |  |
| 杭州_museum | 杭州 | poi_search | False | True | False | None | True | True | False | False |  | 338.46 | 20 | 0 | 1.0000 | 0.0000 | True |  |  |
| 杭州_museum | 杭州 | weather | False | True | False | None | True | True | False | False |  | 311.55 | None | 0 |  |  | True |  |  |
| 杭州_museum | 杭州 | route | False | True | False | None | True | True | False | False | amap | 1349.77 | None | 0 |  |  | True |  |  |
| 杭州_museum | 杭州 | restaurant_search | False | True | False | None | True | True | False | False |  | 225.21 | 4 | 0 | 1.0000 | 0.0000 | True |  |  |
| 杭州_museum | 杭州 | hotel_search | False | True | False | None | True | True | False | False |  | 257.32 | 10 | 0 | 1.0000 | 0.0000 | True |  |  |
| 杭州_museum | 杭州 | budget_estimate | False | True | False | None | True | True | False | False |  | 0.46 | 1 | 0 |  |  | True |  |  |
| 杭州_night_food | 杭州 | poi_search | False | True | False | None | True | True | False | False |  | 259.13 | 20 | 0 | 1.0000 | 0.0000 | True |  |  |
| 杭州_night_food | 杭州 | weather | False | True | False | None | True | True | False | False |  | 224.16 | None | 0 |  |  | True |  |  |
| 杭州_night_food | 杭州 | route | False | True | False | None | True | True | False | False | amap | 623.21 | None | 0 |  |  | True |  |  |
| 杭州_night_food | 杭州 | restaurant_search | False | True | False | None | True | True | False | False |  | 215.99 | 10 | 0 | 1.0000 | 0.0000 | True |  |  |
| 杭州_night_food | 杭州 | hotel_search | False | True | True | primary_unusable | True | False | True | False |  | 86.52 | 0 | 0 |  |  | True |  |  |
| 杭州_night_food | 杭州 | budget_estimate | False | True | False | None | True | True | False | False |  | 0.52 | 1 | 0 |  |  | True |  |  |
| 杭州_route_long | 杭州 | poi_search | False | True | False | None | True | True | False | False |  | 269.5 | 9 | 0 | 1.0000 | 0.0000 | True |  |  |
| 杭州_route_long | 杭州 | weather | False | True | False | None | True | True | False | False |  | 244.06 | None | 0 |  |  | True |  |  |
| 杭州_route_long | 杭州 | route | False | True | False | None | True | True | False | False | amap | 896.74 | None | 0 |  |  | True |  |  |
| 杭州_route_long | 杭州 | restaurant_search | False | True | False | None | True | True | False | False |  | 204.84 | 4 | 0 | 1.0000 | 0.0000 | True |  |  |
| 杭州_route_long | 杭州 | hotel_search | False | True | False | None | True | True | False | False |  | 278.6 | 10 | 0 | 1.0000 | 0.0000 | True |  |  |
| 杭州_route_long | 杭州 | budget_estimate | False | True | False | None | True | True | False | False |  | 0.89 | 1 | 0 |  |  | True |  |  |
| 成都_food | 成都 | poi_search | False | True | False | None | True | True | False | False |  | 466.45 | 20 | 0 | 1.0000 | 0.0000 | True |  |  |
| 成都_food | 成都 | weather | False | True | False | None | True | True | False | False |  | 250.4 | None | 0 |  |  | True |  |  |
| 成都_food | 成都 | route | False | True | False | None | True | True | False | False | amap | 705.51 | None | 0 |  |  | True |  |  |
| 成都_food | 成都 | restaurant_search | False | True | False | None | True | True | False | False |  | 220.05 | 3 | 0 | 1.0000 | 0.0000 | True |  |  |
| 成都_food | 成都 | hotel_search | False | True | False | None | True | True | False | False |  | 283.38 | 10 | 0 | 1.0000 | 0.0000 | True |  |  |
| 成都_food | 成都 | budget_estimate | False | True | False | None | True | True | False | False |  | 0.54 | 1 | 0 |  |  | True |  |  |
| 成都_history | 成都 | poi_search | False | True | True | primary_unusable | True | False | False | False |  | 104.69 | 5 | 0 | 1.0000 | 0.0000 | True |  |  |
| 成都_history | 成都 | weather | False | True | False | None | True | True | False | False |  | 226.9 | None | 0 |  |  | True |  |  |
| 成都_history | 成都 | route | False | True | False | None | True | True | False | False | amap | 793.25 | None | 0 |  |  | True |  |  |
| 成都_history | 成都 | restaurant_search | False | True | False | None | True | True | False | False |  | 223.1 | 2 | 0 | 1.0000 | 0.0000 | True |  |  |
| 成都_history | 成都 | hotel_search | False | True | False | None | True | True | False | False |  | 351.5 | 10 | 0 | 1.0000 | 0.0000 | True |  |  |
| 成都_history | 成都 | budget_estimate | False | True | False | None | True | True | False | False |  | 0.39 | 1 | 0 |  |  | True |  |  |
| 成都_nature | 成都 | poi_search | False | True | False | None | True | True | False | False |  | 498.9 | 20 | 0 | 0.9000 | 0.0000 | True |  |  |
| 成都_nature | 成都 | weather | False | True | False | None | True | True | False | False |  | 285.49 | None | 0 |  |  | True |  |  |
| 成都_nature | 成都 | route | False | True | False | None | True | True | False | False | amap | 990.14 | None | 0 |  |  | True |  |  |
| 成都_nature | 成都 | restaurant_search | False | True | False | None | True | True | False | False |  | 262.67 | 2 | 0 | 1.0000 | 0.0000 | True |  |  |
| 成都_nature | 成都 | hotel_search | False | True | False | None | True | True | False | False |  | 364.87 | 10 | 0 | 1.0000 | 0.0000 | True |  |  |
| 成都_nature | 成都 | budget_estimate | False | True | False | None | True | True | False | False |  | 0.5 | 1 | 0 |  |  | True |  |  |
| 成都_shopping | 成都 | poi_search | False | True | False | None | True | True | False | False |  | 360.95 | 20 | 0 | 0.8500 | 0.0000 | True |  |  |
| 成都_shopping | 成都 | weather | False | True | False | None | True | True | False | False |  | 379.15 | None | 0 |  |  | True |  |  |
| 成都_shopping | 成都 | route | False | True | False | None | True | True | False | False | amap | 1247.41 | None | 0 |  |  | True |  |  |
| 成都_shopping | 成都 | restaurant_search | False | True | False | None | True | True | False | False |  | 313.49 | 2 | 0 | 1.0000 | 0.0000 | True |  |  |
| 成都_shopping | 成都 | hotel_search | False | True | False | None | True | True | False | False |  | 281.13 | 10 | 0 | 1.0000 | 0.0000 | True |  |  |
| 成都_shopping | 成都 | budget_estimate | False | True | False | None | True | True | False | False |  | 0.57 | 1 | 0 |  |  | True |  |  |
| 成都_family | 成都 | poi_search | False | True | False | None | True | True | False | False |  | 189.0 | 8 | 0 | 1.0000 | 0.0000 | True |  |  |
| 成都_family | 成都 | weather | False | True | False | None | True | True | False | False |  | 217.18 | None | 0 |  |  | True |  |  |
| 成都_family | 成都 | route | False | True | False | None | True | True | False | False | amap | 540.26 | None | 0 |  |  | True |  |  |
| 成都_family | 成都 | restaurant_search | False | True | False | None | True | True | False | False |  | 235.7 | 2 | 0 | 1.0000 | 0.0000 | True |  |  |
| 成都_family | 成都 | hotel_search | False | True | False | None | True | True | False | False |  | 335.56 | 10 | 0 | 1.0000 | 0.0000 | True |  |  |
| 成都_family | 成都 | budget_estimate | False | True | False | None | True | True | False | False |  | 0.41 | 1 | 0 |  |  | True |  |  |
| 成都_museum | 成都 | poi_search | False | True | False | None | True | True | False | False |  | 336.45 | 20 | 0 | 0.9500 | 0.0000 | True |  |  |
| 成都_museum | 成都 | weather | False | True | False | None | True | True | False | False |  | 210.9 | None | 0 |  |  | True |  |  |
| 成都_museum | 成都 | route | False | True | False | None | True | True | False | False | amap | 613.04 | None | 0 |  |  | True |  |  |
| 成都_museum | 成都 | restaurant_search | False | True | False | None | True | True | False | False |  | 213.19 | 2 | 0 | 1.0000 | 0.0000 | True |  |  |
| 成都_museum | 成都 | hotel_search | False | True | False | None | True | True | False | False |  | 297.5 | 10 | 0 | 1.0000 | 0.0000 | True |  |  |
| 成都_museum | 成都 | budget_estimate | False | True | False | None | True | True | False | False |  | 0.38 | 1 | 0 |  |  | True |  |  |
| 成都_night_food | 成都 | poi_search | False | True | False | None | True | True | False | False |  | 323.02 | 20 | 0 | 0.9500 | 0.0000 | True |  |  |
| 成都_night_food | 成都 | weather | False | True | False | None | True | True | False | False |  | 207.43 | None | 0 |  |  | True |  |  |
| 成都_night_food | 成都 | route | False | True | False | None | True | True | False | False | amap | 570.01 | None | 0 |  |  | True |  |  |
| 成都_night_food | 成都 | restaurant_search | False | True | False | None | True | True | False | False |  | 245.07 | 10 | 0 | 1.0000 | 0.0000 | True |  |  |
| 成都_night_food | 成都 | hotel_search | False | True | False | None | True | True | False | False |  | 343.91 | 10 | 0 | 1.0000 | 0.0000 | True |  |  |
| 成都_night_food | 成都 | budget_estimate | False | True | False | None | True | True | False | False |  | 0.4 | 1 | 0 |  |  | True |  |  |
| 成都_route_long | 成都 | poi_search | False | True | False | None | True | True | False | False |  | 251.92 | 5 | 0 | 0.8000 | 0.0000 | True |  |  |
| 成都_route_long | 成都 | weather | False | True | False | None | True | True | False | False |  | 217.91 | None | 0 |  |  | True |  |  |
| 成都_route_long | 成都 | route | False | True | False | None | True | True | False | False | amap | 938.79 | None | 0 |  |  | True |  |  |
| 成都_route_long | 成都 | restaurant_search | False | True | False | None | True | True | False | False |  | 224.98 | 2 | 0 | 1.0000 | 0.0000 | True |  |  |
| 成都_route_long | 成都 | hotel_search | False | True | False | None | True | True | False | False |  | 253.07 | 10 | 0 | 1.0000 | 0.0000 | True |  |  |
| 成都_route_long | 成都 | budget_estimate | False | True | False | None | True | True | False | False |  | 0.41 | 1 | 0 |  |  | True |  |  |
| 广州_food | 广州 | poi_search | False | True | False | None | True | True | False | False |  | 479.63 | 20 | 0 | 1.0000 | 0.0000 | True |  |  |
| 广州_food | 广州 | weather | False | True | False | None | True | True | False | False |  | 270.05 | None | 0 |  |  | True |  |  |
| 广州_food | 广州 | route | False | True | False | None | True | True | False | False | amap | 765.14 | None | 0 |  |  | True |  |  |
| 广州_food | 广州 | restaurant_search | False | True | False | None | True | True | False | False |  | 281.45 | 10 | 0 | 1.0000 | 0.0000 | True |  |  |
| 广州_food | 广州 | hotel_search | False | True | False | None | True | True | False | False |  | 288.95 | 10 | 0 | 1.0000 | 0.0000 | True |  |  |
| 广州_food | 广州 | budget_estimate | False | True | False | None | True | True | False | False |  | 0.42 | 1 | 0 |  |  | True |  |  |
| 广州_history | 广州 | poi_search | False | True | False | None | True | True | False | False |  | 196.27 | 2 | 0 | 1.0000 | 0.0000 | True |  |  |
| 广州_history | 广州 | weather | False | True | False | None | True | True | False | False |  | 229.96 | None | 0 |  |  | True |  |  |
| 广州_history | 广州 | route | False | True | False | None | True | True | False | False | amap | 817.71 | None | 0 |  |  | True |  |  |
| 广州_history | 广州 | restaurant_search | False | True | False | None | True | True | False | False |  | 260.85 | 8 | 0 | 1.0000 | 0.0000 | True |  |  |
| 广州_history | 广州 | hotel_search | False | True | False | None | True | True | False | False |  | 327.09 | 10 | 0 | 1.0000 | 0.0000 | True |  |  |
| 广州_history | 广州 | budget_estimate | False | True | False | None | True | True | False | False |  | 0.55 | 1 | 0 |  |  | True |  |  |
| 广州_nature | 广州 | poi_search | False | True | False | None | True | True | False | False |  | 1019.42 | 20 | 0 | 1.0000 | 0.0000 | True |  |  |
| 广州_nature | 广州 | weather | False | True | False | None | True | True | False | False |  | 225.5 | None | 0 |  |  | True |  |  |
| 广州_nature | 广州 | route | False | True | False | None | True | True | False | False | amap | 1154.22 | None | 0 |  |  | True |  |  |
| 广州_nature | 广州 | restaurant_search | False | True | False | None | True | True | False | False |  | 256.81 | 8 | 0 | 1.0000 | 0.0000 | True |  |  |
| 广州_nature | 广州 | hotel_search | False | True | False | None | True | True | False | False |  | 318.68 | 10 | 0 | 1.0000 | 0.0000 | True |  |  |
| 广州_nature | 广州 | budget_estimate | False | True | False | None | True | True | False | False |  | 0.48 | 1 | 0 |  |  | True |  |  |
| 广州_shopping | 广州 | poi_search | False | True | False | None | True | True | False | False |  | 291.44 | 20 | 0 | 0.8500 | 0.0000 | True |  |  |
| 广州_shopping | 广州 | weather | False | True | False | None | True | True | False | False |  | 205.92 | None | 0 |  |  | True |  |  |
| 广州_shopping | 广州 | route | False | True | False | None | True | True | False | False | amap | 556.58 | None | 0 |  |  | True |  |  |
| 广州_shopping | 广州 | restaurant_search | False | True | False | None | True | True | False | False |  | 285.14 | 8 | 0 | 1.0000 | 0.0000 | True |  |  |
| 广州_shopping | 广州 | hotel_search | False | True | False | None | True | True | False | False |  | 269.43 | 10 | 0 | 1.0000 | 0.0000 | True |  |  |
| 广州_shopping | 广州 | budget_estimate | False | True | False | None | True | True | False | False |  | 0.4 | 1 | 0 |  |  | True |  |  |
| 广州_family | 广州 | poi_search | False | True | False | None | True | True | False | False |  | 195.72 | 7 | 0 | 1.0000 | 0.0000 | True |  |  |
| 广州_family | 广州 | weather | False | True | False | None | True | True | False | False |  | 206.29 | None | 0 |  |  | True |  |  |
| 广州_family | 广州 | route | False | True | False | None | True | True | False | False | amap | 682.33 | None | 0 |  |  | True |  |  |
| 广州_family | 广州 | restaurant_search | False | True | False | None | True | True | False | False |  | 268.54 | 8 | 0 | 1.0000 | 0.0000 | True |  |  |
| 广州_family | 广州 | hotel_search | False | True | False | None | True | True | False | False |  | 314.77 | 10 | 0 | 1.0000 | 0.0000 | True |  |  |
| 广州_family | 广州 | budget_estimate | False | True | False | None | True | True | False | False |  | 0.57 | 1 | 0 |  |  | True |  |  |
| 广州_museum | 广州 | poi_search | False | True | False | None | True | True | False | False |  | 334.09 | 20 | 0 | 1.0000 | 0.0000 | True |  |  |
| 广州_museum | 广州 | weather | False | True | False | None | True | True | False | False |  | 241.25 | None | 0 |  |  | True |  |  |
| 广州_museum | 广州 | route | False | True | False | None | True | True | False | False | amap | 594.71 | None | 0 |  |  | True |  |  |
| 广州_museum | 广州 | restaurant_search | False | True | False | None | True | True | False | False |  | 240.79 | 8 | 0 | 1.0000 | 0.0000 | True |  |  |
| 广州_museum | 广州 | hotel_search | False | True | True | primary_unusable | True | False | True | False |  | 105.02 | 0 | 0 |  |  | True |  |  |
| 广州_museum | 广州 | budget_estimate | False | True | False | None | True | True | False | False |  | 0.43 | 1 | 0 |  |  | True |  |  |
| 广州_night_food | 广州 | poi_search | False | True | False | None | True | True | False | False |  | 283.95 | 20 | 0 | 1.0000 | 0.0000 | True |  |  |
| 广州_night_food | 广州 | weather | False | True | False | None | True | True | False | False |  | 224.82 | None | 0 |  |  | True |  |  |
| 广州_night_food | 广州 | route | False | True | False | None | True | True | False | False | amap | 622.06 | None | 0 |  |  | True |  |  |
| 广州_night_food | 广州 | restaurant_search | False | True | False | None | True | True | False | False |  | 232.46 | 10 | 0 | 1.0000 | 0.0000 | True |  |  |
| 广州_night_food | 广州 | hotel_search | False | True | False | None | True | True | False | False |  | 311.19 | 10 | 0 | 1.0000 | 0.0000 | True |  |  |
| 广州_night_food | 广州 | budget_estimate | False | True | False | None | True | True | False | False |  | 0.73 | 1 | 0 |  |  | True |  |  |
| 广州_route_long | 广州 | poi_search | False | True | False | None | True | True | False | False |  | 295.79 | 10 | 0 | 1.0000 | 0.0000 | True |  |  |
| 广州_route_long | 广州 | weather | False | True | False | None | True | True | False | False |  | 228.02 | None | 0 |  |  | True |  |  |
| 广州_route_long | 广州 | route | False | True | False | None | True | True | False | False | amap | 877.87 | None | 0 |  |  | True |  |  |
| 广州_route_long | 广州 | restaurant_search | False | True | False | None | True | True | False | False |  | 240.92 | 8 | 0 | 1.0000 | 0.0000 | True |  |  |
| 广州_route_long | 广州 | hotel_search | False | True | False | None | True | True | False | False |  | 266.75 | 10 | 0 | 1.0000 | 0.0000 | True |  |  |
| 广州_route_long | 广州 | budget_estimate | False | True | False | None | True | True | False | False |  | 0.56 | 1 | 0 |  |  | True |  |  |
| 深圳_food | 深圳 | poi_search | False | True | False | None | True | True | False | False |  | 503.41 | 20 | 0 | 1.0000 | 0.0000 | True |  |  |
| 深圳_food | 深圳 | weather | False | True | False | None | True | True | False | False |  | 293.91 | None | 0 |  |  | True |  |  |
| 深圳_food | 深圳 | route | False | True | False | None | True | True | False | False | amap | 1271.23 | None | 0 |  |  | True |  |  |
| 深圳_food | 深圳 | restaurant_search | False | True | False | None | True | True | False | False |  | 356.88 | 10 | 0 | 1.0000 | 0.0000 | True |  |  |
| 深圳_food | 深圳 | hotel_search | False | True | False | None | True | True | False | False |  | 532.04 | 10 | 0 | 1.0000 | 0.0000 | True |  |  |
| 深圳_food | 深圳 | budget_estimate | False | True | False | None | True | True | False | False |  | 0.7 | 1 | 0 |  |  | True |  |  |
| 深圳_history | 深圳 | poi_search | False | True | False | None | True | True | False | False |  | 229.9 | 6 | 0 | 0.6667 | 0.0000 | True |  |  |
| 深圳_history | 深圳 | weather | False | True | False | None | True | True | False | False |  | 209.88 | None | 0 |  |  | True |  |  |
| 深圳_history | 深圳 | route | False | True | False | None | True | True | False | False | amap | 921.65 | None | 0 |  |  | True |  |  |
| 深圳_history | 深圳 | restaurant_search | False | True | False | None | True | True | False | False |  | 237.05 | 6 | 0 | 1.0000 | 0.0000 | True |  |  |
| 深圳_history | 深圳 | hotel_search | False | True | False | None | True | True | False | False |  | 299.78 | 10 | 0 | 1.0000 | 0.0000 | True |  |  |
| 深圳_history | 深圳 | budget_estimate | False | True | False | None | True | True | False | False |  | 0.59 | 1 | 0 |  |  | True |  |  |
| 深圳_nature | 深圳 | poi_search | False | True | False | None | True | True | False | False |  | 423.19 | 20 | 0 | 0.9500 | 0.0000 | True |  |  |
| 深圳_nature | 深圳 | weather | False | True | False | None | True | True | False | False |  | 216.38 | None | 0 |  |  | True |  |  |
| 深圳_nature | 深圳 | route | False | True | False | None | True | True | False | False | amap | 958.87 | None | 0 |  |  | True |  |  |
| 深圳_nature | 深圳 | restaurant_search | False | True | False | None | True | True | False | False |  | 232.85 | 6 | 0 | 1.0000 | 0.0000 | True |  |  |
| 深圳_nature | 深圳 | hotel_search | False | True | False | None | True | True | False | False |  | 291.4 | 10 | 0 | 1.0000 | 0.0000 | True |  |  |
| 深圳_nature | 深圳 | budget_estimate | False | True | False | None | True | True | False | False |  | 0.57 | 1 | 0 |  |  | True |  |  |
| 深圳_shopping | 深圳 | poi_search | False | True | False | None | True | True | False | False |  | 315.63 | 20 | 0 | 1.0000 | 0.0000 | True |  |  |
| 深圳_shopping | 深圳 | weather | False | True | False | None | True | True | False | False |  | 221.58 | None | 0 |  |  | True |  |  |
| 深圳_shopping | 深圳 | route | False | True | False | None | True | True | False | False | amap | 594.98 | None | 0 |  |  | True |  |  |
| 深圳_shopping | 深圳 | restaurant_search | False | True | False | None | True | True | False | False |  | 231.03 | 6 | 0 | 1.0000 | 0.0000 | True |  |  |
| 深圳_shopping | 深圳 | hotel_search | False | True | True | primary_unusable | True | False | True | False |  | 126.43 | 0 | 0 |  |  | True |  |  |
| 深圳_shopping | 深圳 | budget_estimate | False | True | False | None | True | True | False | False |  | 0.52 | 1 | 0 |  |  | True |  |  |
| 深圳_family | 深圳 | poi_search | False | True | False | None | True | True | False | False |  | 238.57 | 16 | 0 | 0.8750 | 0.0000 | True |  |  |
| 深圳_family | 深圳 | weather | False | True | False | None | True | True | False | False |  | 219.3 | None | 0 |  |  | True |  |  |
| 深圳_family | 深圳 | route | False | True | True | primary_unusable | True | False | True | True |  | 291.35 | None | 0 |  |  | True |  |  |
| 深圳_family | 深圳 | restaurant_search | False | True | True | primary_unusable | True | False | True | False |  | 128.41 | 0 | 0 |  |  | True |  |  |
| 深圳_family | 深圳 | hotel_search | False | True | True | primary_unusable | True | False | True | False |  | 89.21 | 0 | 0 |  |  | True |  |  |
| 深圳_family | 深圳 | budget_estimate | False | True | False | None | True | True | False | False |  | 0.48 | 1 | 0 |  |  | True |  |  |
| 深圳_museum | 深圳 | poi_search | False | True | False | None | True | True | False | False |  | 316.43 | 20 | 0 | 1.0000 | 0.0000 | True |  |  |
| 深圳_museum | 深圳 | weather | False | True | False | None | True | True | False | False |  | 224.77 | None | 0 |  |  | True |  |  |
| 深圳_museum | 深圳 | route | False | True | False | None | True | True | False | False | amap | 642.86 | None | 0 |  |  | True |  |  |
| 深圳_museum | 深圳 | restaurant_search | False | True | False | None | True | True | False | False |  | 224.34 | 6 | 0 | 1.0000 | 0.0000 | True |  |  |
| 深圳_museum | 深圳 | hotel_search | False | True | False | None | True | True | False | False |  | 281.6 | 10 | 0 | 1.0000 | 0.0000 | True |  |  |
| 深圳_museum | 深圳 | budget_estimate | False | True | False | None | True | True | False | False |  | 0.91 | 1 | 0 |  |  | True |  |  |
| 深圳_night_food | 深圳 | poi_search | False | True | False | None | True | True | False | False |  | 305.63 | 20 | 0 | 1.0000 | 0.0000 | True |  |  |
| 深圳_night_food | 深圳 | weather | False | True | False | None | True | True | False | False |  | 286.92 | None | 0 |  |  | True |  |  |
| 深圳_night_food | 深圳 | route | False | True | False | None | True | True | False | False | amap | 613.88 | None | 0 |  |  | True |  |  |
| 深圳_night_food | 深圳 | restaurant_search | False | True | False | None | True | True | False | False |  | 250.78 | 10 | 0 | 1.0000 | 0.0000 | True |  |  |
| 深圳_night_food | 深圳 | hotel_search | False | True | False | None | True | True | False | False |  | 311.52 | 10 | 0 | 1.0000 | 0.0000 | True |  |  |
| 深圳_night_food | 深圳 | budget_estimate | False | True | False | None | True | True | False | False |  | 0.44 | 1 | 0 |  |  | True |  |  |
| 深圳_route_long | 深圳 | poi_search | False | True | False | None | True | True | False | False |  | 269.66 | 5 | 0 | 0.8000 | 0.0000 | True |  |  |
| 深圳_route_long | 深圳 | weather | False | True | False | None | True | True | False | False |  | 918.09 | None | 0 |  |  | True |  |  |
| 深圳_route_long | 深圳 | route | False | True | False | None | True | True | False | False | amap | 870.22 | None | 0 |  |  | True |  |  |
| 深圳_route_long | 深圳 | restaurant_search | False | True | False | None | True | True | False | False |  | 261.0 | 6 | 0 | 1.0000 | 0.0000 | True |  |  |
| 深圳_route_long | 深圳 | hotel_search | False | True | False | None | True | True | False | False |  | 277.14 | 10 | 0 | 1.0000 | 0.0000 | True |  |  |
| 深圳_route_long | 深圳 | budget_estimate | False | True | False | None | True | True | False | False |  | 1.2 | 1 | 0 |  |  | True |  |  |
| 重庆_food | 重庆 | poi_search | False | True | False | None | True | True | False | False |  | 480.1 | 20 | 0 | 1.0000 | 0.0000 | True |  |  |
| 重庆_food | 重庆 | weather | False | True | False | None | True | True | False | False |  | 232.81 | None | 0 |  |  | True |  |  |
| 重庆_food | 重庆 | route | False | True | False | None | True | True | False | False | amap | 739.38 | None | 0 |  |  | True |  |  |
| 重庆_food | 重庆 | restaurant_search | False | True | False | None | True | True | False | False |  | 237.92 | 5 | 0 | 1.0000 | 0.0000 | True |  |  |
| 重庆_food | 重庆 | hotel_search | False | True | False | None | True | True | False | False |  | 311.08 | 10 | 0 | 1.0000 | 0.0000 | True |  |  |
| 重庆_food | 重庆 | budget_estimate | False | True | False | None | True | True | False | False |  | 0.77 | 1 | 0 |  |  | True |  |  |
| 重庆_history | 重庆 | poi_search | False | True | False | None | True | True | False | False |  | 170.01 | 2 | 0 | 1.0000 | 0.0000 | True |  |  |
| 重庆_history | 重庆 | weather | False | True | False | None | True | True | False | False |  | 222.81 | None | 0 |  |  | True |  |  |
| 重庆_history | 重庆 | route | False | True | False | None | True | True | False | False | amap | 796.36 | None | 0 |  |  | True |  |  |
| 重庆_history | 重庆 | restaurant_search | False | True | False | None | True | True | False | False |  | 223.86 | 1 | 0 | 1.0000 | 0.0000 | True |  |  |
| 重庆_history | 重庆 | hotel_search | False | True | False | None | True | True | False | False |  | 328.41 | 10 | 0 | 1.0000 | 0.0000 | True |  |  |
| 重庆_history | 重庆 | budget_estimate | False | True | False | None | True | True | False | False |  | 0.71 | 1 | 0 |  |  | True |  |  |
| 重庆_nature | 重庆 | poi_search | False | True | False | None | True | True | False | False |  | 457.55 | 20 | 0 | 0.9500 | 0.0000 | True |  |  |
| 重庆_nature | 重庆 | weather | False | True | False | None | True | True | False | False |  | 209.78 | None | 0 |  |  | True |  |  |
| 重庆_nature | 重庆 | route | False | True | False | None | True | True | False | False | amap | 882.12 | None | 0 |  |  | True |  |  |
| 重庆_nature | 重庆 | restaurant_search | False | True | False | None | True | True | False | False |  | 200.56 | 1 | 0 | 1.0000 | 0.0000 | True |  |  |
| 重庆_nature | 重庆 | hotel_search | False | True | False | None | True | True | False | False |  | 284.3 | 10 | 0 | 1.0000 | 0.0000 | True |  |  |
| 重庆_nature | 重庆 | budget_estimate | False | True | False | None | True | True | False | False |  | 0.8 | 1 | 0 |  |  | True |  |  |
| 重庆_shopping | 重庆 | poi_search | False | True | False | None | True | True | False | False |  | 444.43 | 20 | 0 | 1.0000 | 0.0000 | True |  |  |
| 重庆_shopping | 重庆 | weather | False | True | False | None | True | True | False | False |  | 238.16 | None | 0 |  |  | True |  |  |
| 重庆_shopping | 重庆 | route | False | True | False | None | True | True | False | False | amap | 604.14 | None | 0 |  |  | True |  |  |
| 重庆_shopping | 重庆 | restaurant_search | False | True | False | None | True | True | False | False |  | 208.71 | 1 | 0 | 1.0000 | 0.0000 | True |  |  |
| 重庆_shopping | 重庆 | hotel_search | False | True | False | None | True | True | False | False |  | 284.16 | 10 | 0 | 1.0000 | 0.0000 | True |  |  |
| 重庆_shopping | 重庆 | budget_estimate | False | True | False | None | True | True | False | False |  | 0.65 | 1 | 0 |  |  | True |  |  |
| 重庆_family | 重庆 | poi_search | False | True | False | None | True | True | False | False |  | 202.17 | 5 | 0 | 1.0000 | 0.0000 | True |  |  |
| 重庆_family | 重庆 | weather | False | True | False | None | True | True | False | False |  | 242.07 | None | 0 |  |  | True |  |  |
| 重庆_family | 重庆 | route | False | True | False | None | True | True | False | False | amap | 640.14 | None | 0 |  |  | True |  |  |
| 重庆_family | 重庆 | restaurant_search | False | True | False | None | True | True | False | False |  | 206.8 | 1 | 0 | 1.0000 | 0.0000 | True |  |  |
| 重庆_family | 重庆 | hotel_search | False | True | True | primary_unusable | True | False | True | False |  | 81.72 | 0 | 0 |  |  | True |  |  |
| 重庆_family | 重庆 | budget_estimate | False | True | False | None | True | True | False | False |  | 0.45 | 1 | 0 |  |  | True |  |  |
| 重庆_museum | 重庆 | poi_search | False | True | False | None | True | True | False | False |  | 347.58 | 20 | 0 | 1.0000 | 0.0000 | True |  |  |
| 重庆_museum | 重庆 | weather | False | True | False | None | True | True | False | False |  | 332.24 | None | 0 |  |  | True |  |  |
| 重庆_museum | 重庆 | route | False | True | False | None | True | True | False | False | amap | 1171.77 | None | 0 |  |  | True |  |  |
| 重庆_museum | 重庆 | restaurant_search | False | True | False | None | True | True | False | False |  | 212.92 | 1 | 0 | 1.0000 | 0.0000 | True |  |  |
| 重庆_museum | 重庆 | hotel_search | False | True | False | None | True | True | False | False |  | 337.43 | 10 | 0 | 1.0000 | 0.0000 | True |  |  |
| 重庆_museum | 重庆 | budget_estimate | False | True | False | None | True | True | False | False |  | 0.48 | 1 | 0 |  |  | True |  |  |
| 重庆_night_food | 重庆 | poi_search | False | True | False | None | True | True | False | False |  | 290.56 | 20 | 0 | 1.0000 | 0.0000 | True |  |  |
| 重庆_night_food | 重庆 | weather | False | True | False | None | True | True | False | False |  | 216.62 | None | 0 |  |  | True |  |  |
| 重庆_night_food | 重庆 | route | False | True | False | None | True | True | False | False | amap | 557.84 | None | 0 |  |  | True |  |  |
| 重庆_night_food | 重庆 | restaurant_search | False | True | False | None | True | True | False | False |  | 250.59 | 10 | 0 | 1.0000 | 0.0000 | True |  |  |
| 重庆_night_food | 重庆 | hotel_search | False | True | False | None | True | True | False | False |  | 300.48 | 10 | 0 | 1.0000 | 0.0000 | True |  |  |
| 重庆_night_food | 重庆 | budget_estimate | False | True | False | None | True | True | False | False |  | 0.73 | 1 | 0 |  |  | True |  |  |
| 重庆_route_long | 重庆 | poi_search | False | True | False | None | True | True | False | False |  | 445.68 | 13 | 0 | 1.0000 | 0.0000 | True |  |  |
| 重庆_route_long | 重庆 | weather | False | True | False | None | True | True | False | False |  | 220.34 | None | 0 |  |  | True |  |  |
| 重庆_route_long | 重庆 | route | False | True | False | None | True | True | False | False | amap | 1137.43 | None | 0 |  |  | True |  |  |
| 重庆_route_long | 重庆 | restaurant_search | False | True | False | None | True | True | False | False |  | 227.89 | 1 | 0 | 1.0000 | 0.0000 | True |  |  |
| 重庆_route_long | 重庆 | hotel_search | False | True | False | None | True | True | False | False |  | 278.3 | 10 | 0 | 1.0000 | 0.0000 | True |  |  |
| 重庆_route_long | 重庆 | budget_estimate | False | True | False | None | True | True | False | False |  | 0.51 | 1 | 0 |  |  | True |  |  |
| 南京_food | 南京 | poi_search | False | True | False | None | True | True | False | False |  | 459.97 | 20 | 0 | 1.0000 | 0.0000 | True |  |  |
| 南京_food | 南京 | weather | False | True | False | None | True | True | False | False |  | 223.42 | None | 0 |  |  | True |  |  |
| 南京_food | 南京 | route | False | True | False | None | True | True | False | False | amap | 670.18 | None | 0 |  |  | True |  |  |
| 南京_food | 南京 | restaurant_search | False | True | False | None | True | True | False | False |  | 259.38 | 5 | 0 | 1.0000 | 0.0000 | True |  |  |
| 南京_food | 南京 | hotel_search | False | True | False | None | True | True | False | False |  | 272.89 | 10 | 0 | 1.0000 | 0.0000 | True |  |  |
| 南京_food | 南京 | budget_estimate | False | True | False | None | True | True | False | False |  | 0.82 | 1 | 0 |  |  | True |  |  |
| 南京_history | 南京 | poi_search | False | True | False | None | True | True | False | False |  | 196.85 | 2 | 0 | 1.0000 | 0.0000 | True |  |  |
| 南京_history | 南京 | weather | False | True | False | None | True | True | False | False |  | 240.98 | None | 0 |  |  | True |  |  |
| 南京_history | 南京 | route | False | True | True | primary_unusable | True | False | True | True |  | 299.08 | None | 0 |  |  | True |  |  |
| 南京_history | 南京 | restaurant_search | False | True | False | None | True | True | False | False |  | 217.49 | 2 | 0 | 1.0000 | 0.0000 | True |  |  |
| 南京_history | 南京 | hotel_search | False | True | False | None | True | True | False | False |  | 269.8 | 10 | 0 | 1.0000 | 0.0000 | True |  |  |
| 南京_history | 南京 | budget_estimate | False | True | False | None | True | True | False | False |  | 0.42 | 1 | 0 |  |  | True |  |  |
| 南京_nature | 南京 | poi_search | False | True | False | None | True | True | False | False |  | 379.91 | 20 | 0 | 1.0000 | 0.0000 | True |  |  |
| 南京_nature | 南京 | weather | False | True | False | None | True | True | False | False |  | 230.43 | None | 0 |  |  | True |  |  |
| 南京_nature | 南京 | route | False | True | False | None | True | True | False | False | amap | 999.99 | None | 0 |  |  | True |  |  |
| 南京_nature | 南京 | restaurant_search | False | True | False | None | True | True | False | False |  | 231.14 | 2 | 0 | 1.0000 | 0.0000 | True |  |  |
| 南京_nature | 南京 | hotel_search | False | True | False | None | True | True | False | False |  | 242.26 | 10 | 0 | 1.0000 | 0.0000 | True |  |  |
| 南京_nature | 南京 | budget_estimate | False | True | False | None | True | True | False | False |  | 0.76 | 1 | 0 |  |  | True |  |  |
| 南京_shopping | 南京 | poi_search | False | True | False | None | True | True | False | False |  | 286.54 | 20 | 0 | 0.9000 | 0.0000 | True |  |  |
| 南京_shopping | 南京 | weather | False | True | False | None | True | True | False | False |  | 214.07 | None | 0 |  |  | True |  |  |
| 南京_shopping | 南京 | route | False | True | False | None | True | True | False | False | amap | 638.15 | None | 0 |  |  | True |  |  |
| 南京_shopping | 南京 | restaurant_search | False | True | False | None | True | True | False | False |  | 247.56 | 2 | 0 | 1.0000 | 0.0000 | True |  |  |
| 南京_shopping | 南京 | hotel_search | False | True | False | None | True | True | False | False |  | 381.06 | 10 | 0 | 1.0000 | 0.0000 | True |  |  |
| 南京_shopping | 南京 | budget_estimate | False | True | False | None | True | True | False | False |  | 0.49 | 1 | 0 |  |  | True |  |  |
| 南京_family | 南京 | poi_search | False | True | False | None | True | True | False | False |  | 421.34 | 3 | 0 | 1.0000 | 0.0000 | True |  |  |
| 南京_family | 南京 | weather | False | True | False | None | True | True | False | False |  | 442.58 | None | 0 |  |  | True |  |  |
| 南京_family | 南京 | route | False | True | False | None | True | True | False | False | amap | 626.97 | None | 0 |  |  | True |  |  |
| 南京_family | 南京 | restaurant_search | False | True | False | None | True | True | False | False |  | 193.99 | 2 | 0 | 1.0000 | 0.0000 | True |  |  |
| 南京_family | 南京 | hotel_search | False | True | False | None | True | True | False | False |  | 265.33 | 10 | 0 | 1.0000 | 0.0000 | True |  |  |
| 南京_family | 南京 | budget_estimate | False | True | False | None | True | True | False | False |  | 0.61 | 1 | 0 |  |  | True |  |  |
| 南京_museum | 南京 | poi_search | False | True | False | None | True | True | False | False |  | 314.67 | 20 | 0 | 1.0000 | 0.0000 | True |  |  |
| 南京_museum | 南京 | weather | False | True | False | None | True | True | False | False |  | 220.35 | None | 0 |  |  | True |  |  |
| 南京_museum | 南京 | route | False | True | False | None | True | True | False | False | amap | 553.72 | None | 0 |  |  | True |  |  |
| 南京_museum | 南京 | restaurant_search | False | True | False | None | True | True | False | False |  | 229.39 | 2 | 0 | 1.0000 | 0.0000 | True |  |  |
| 南京_museum | 南京 | hotel_search | False | True | False | None | True | True | False | False |  | 308.72 | 10 | 0 | 1.0000 | 0.0000 | True |  |  |
| 南京_museum | 南京 | budget_estimate | False | True | False | None | True | True | False | False |  | 0.55 | 1 | 0 |  |  | True |  |  |
| 南京_night_food | 南京 | poi_search | False | True | True | primary_unusable | True | False | True | False |  | 91.59 | 0 | 0 |  |  | True |  |  |
| 南京_night_food | 南京 | weather | False | True | False | None | True | True | False | False |  | 225.87 | None | 0 |  |  | True |  |  |
| 南京_night_food | 南京 | route | False | True | False | None | True | True | False | False | amap | 591.68 | None | 0 |  |  | True |  |  |
| 南京_night_food | 南京 | restaurant_search | False | True | False | None | True | True | False | False |  | 261.94 | 10 | 0 | 1.0000 | 0.0000 | True |  |  |
| 南京_night_food | 南京 | hotel_search | False | True | False | None | True | True | False | False |  | 264.53 | 10 | 0 | 1.0000 | 0.0000 | True |  |  |
| 南京_night_food | 南京 | budget_estimate | False | True | False | None | True | True | False | False |  | 0.46 | 1 | 0 |  |  | True |  |  |
| 南京_route_long | 南京 | poi_search | False | True | False | None | True | True | False | False |  | 461.91 | 11 | 0 | 1.0000 | 0.0000 | True |  |  |
| 南京_route_long | 南京 | weather | False | True | False | None | True | True | False | False |  | 235.01 | None | 0 |  |  | True |  |  |
| 南京_route_long | 南京 | route | False | True | False | None | True | True | False | False | amap | 992.27 | None | 0 |  |  | True |  |  |
| 南京_route_long | 南京 | restaurant_search | False | True | False | None | True | True | False | False |  | 192.3 | 2 | 0 | 1.0000 | 0.0000 | True |  |  |
| 南京_route_long | 南京 | hotel_search | False | True | False | None | True | True | False | False |  | 275.59 | 10 | 0 | 1.0000 | 0.0000 | True |  |  |
| 南京_route_long | 南京 | budget_estimate | False | True | False | None | True | True | False | False |  | 0.48 | 1 | 0 |  |  | True |  |  |
| 苏州_food | 苏州 | poi_search | False | True | False | None | True | True | False | False |  | 429.99 | 20 | 0 | 1.0000 | 0.0000 | True |  |  |
| 苏州_food | 苏州 | weather | False | True | False | None | True | True | False | False |  | 215.08 | None | 0 |  |  | True |  |  |
| 苏州_food | 苏州 | route | False | True | False | None | True | True | False | False | amap | 674.89 | None | 0 |  |  | True |  |  |
| 苏州_food | 苏州 | restaurant_search | False | True | False | None | True | True | False | False |  | 274.74 | 9 | 0 | 1.0000 | 0.0000 | True |  |  |
| 苏州_food | 苏州 | hotel_search | False | True | False | None | True | True | False | False |  | 312.89 | 10 | 0 | 1.0000 | 0.0000 | True |  |  |
| 苏州_food | 苏州 | budget_estimate | False | True | False | None | True | True | False | False |  | 0.5 | 1 | 0 |  |  | True |  |  |
| 苏州_history | 苏州 | poi_search | False | True | False | None | True | True | False | False |  | 297.05 | 20 | 0 | 0.8500 | 0.0000 | True |  |  |
| 苏州_history | 苏州 | weather | False | True | False | None | True | True | False | False |  | 257.53 | None | 0 |  |  | True |  |  |
| 苏州_history | 苏州 | route | False | True | False | None | True | True | False | False | amap | 701.55 | None | 0 |  |  | True |  |  |
| 苏州_history | 苏州 | restaurant_search | False | True | False | None | True | True | False | False |  | 201.93 | 2 | 0 | 1.0000 | 0.0000 | True |  |  |
| 苏州_history | 苏州 | hotel_search | False | True | True | primary_unusable | True | False | True | False |  | 104.83 | 0 | 0 |  |  | True |  |  |
| 苏州_history | 苏州 | budget_estimate | False | True | False | None | True | True | False | False |  | 0.86 | 1 | 0 |  |  | True |  |  |
| 苏州_nature | 苏州 | poi_search | False | True | False | None | True | True | False | False |  | 426.53 | 20 | 0 | 0.9500 | 0.0000 | True |  |  |
| 苏州_nature | 苏州 | weather | False | True | False | None | True | True | False | False |  | 252.65 | None | 0 |  |  | True |  |  |
| 苏州_nature | 苏州 | route | False | True | False | None | True | True | False | False | amap | 1314.03 | None | 0 |  |  | True |  |  |
| 苏州_nature | 苏州 | restaurant_search | False | True | False | None | True | True | False | False |  | 451.8 | 2 | 0 | 1.0000 | 0.0000 | True |  |  |
| 苏州_nature | 苏州 | hotel_search | False | True | False | None | True | True | False | False |  | 292.21 | 10 | 0 | 1.0000 | 0.0000 | True |  |  |
| 苏州_nature | 苏州 | budget_estimate | False | True | False | None | True | True | False | False |  | 0.78 | 1 | 0 |  |  | True |  |  |
| 苏州_shopping | 苏州 | poi_search | False | True | False | None | True | True | False | False |  | 316.17 | 20 | 0 | 1.0000 | 0.0000 | True |  |  |
| 苏州_shopping | 苏州 | weather | False | True | False | None | True | True | False | False |  | 243.67 | None | 0 |  |  | True |  |  |
| 苏州_shopping | 苏州 | route | False | True | False | None | True | True | False | False | amap | 616.62 | None | 0 |  |  | True |  |  |
| 苏州_shopping | 苏州 | restaurant_search | False | True | False | None | True | True | False | False |  | 181.82 | 2 | 0 | 1.0000 | 0.0000 | True |  |  |
| 苏州_shopping | 苏州 | hotel_search | False | True | False | None | True | True | False | False |  | 307.51 | 10 | 0 | 1.0000 | 0.0000 | True |  |  |
| 苏州_shopping | 苏州 | budget_estimate | False | True | False | None | True | True | False | False |  | 0.84 | 1 | 0 |  |  | True |  |  |
| 苏州_family | 苏州 | poi_search | False | True | False | None | True | True | False | False |  | 177.84 | 5 | 0 | 0.8000 | 0.0000 | True |  |  |
| 苏州_family | 苏州 | weather | False | True | False | None | True | True | False | False |  | 231.4 | None | 0 |  |  | True |  |  |
| 苏州_family | 苏州 | route | False | True | False | None | True | True | False | False | amap | 601.25 | None | 0 |  |  | True |  |  |
| 苏州_family | 苏州 | restaurant_search | False | True | False | None | True | True | False | False |  | 217.8 | 2 | 0 | 1.0000 | 0.0000 | True |  |  |
| 苏州_family | 苏州 | hotel_search | False | True | False | None | True | True | False | False |  | 242.52 | 10 | 0 | 1.0000 | 0.0000 | True |  |  |
| 苏州_family | 苏州 | budget_estimate | False | True | False | None | True | True | False | False |  | 0.41 | 1 | 0 |  |  | True |  |  |
| 苏州_museum | 苏州 | poi_search | False | True | True | primary_unusable | True | False | True | False |  | 88.79 | 0 | 0 |  |  | True |  |  |
| 苏州_museum | 苏州 | weather | False | True | False | None | True | True | False | False |  | 217.9 | None | 0 |  |  | True |  |  |
| 苏州_museum | 苏州 | route | False | True | False | None | True | True | False | False | amap | 536.25 | None | 0 |  |  | True |  |  |
| 苏州_museum | 苏州 | restaurant_search | False | True | False | None | True | True | False | False |  | 210.28 | 2 | 0 | 1.0000 | 0.0000 | True |  |  |
| 苏州_museum | 苏州 | hotel_search | False | True | True | primary_unusable | True | False | True | False |  | 128.47 | 0 | 0 |  |  | True |  |  |
| 苏州_museum | 苏州 | budget_estimate | False | True | False | None | True | True | False | False |  | 0.42 | 1 | 0 |  |  | True |  |  |
| 苏州_night_food | 苏州 | poi_search | False | True | False | None | True | True | False | False |  | 249.18 | 20 | 0 | 0.9500 | 0.0000 | True |  |  |
| 苏州_night_food | 苏州 | weather | False | True | False | None | True | True | False | False |  | 245.35 | None | 0 |  |  | True |  |  |
| 苏州_night_food | 苏州 | route | False | True | True | primary_unusable | True | False | True | True |  | 283.75 | None | 0 |  |  | True |  |  |
| 苏州_night_food | 苏州 | restaurant_search | False | True | True | primary_unusable | True | False | True | False |  | 87.79 | 0 | 0 |  |  | True |  |  |
| 苏州_night_food | 苏州 | hotel_search | False | True | True | primary_unusable | True | False | True | False |  | 87.53 | 0 | 0 |  |  | True |  |  |
| 苏州_night_food | 苏州 | budget_estimate | False | True | False | None | True | True | False | False |  | 0.47 | 1 | 0 |  |  | True |  |  |
| 苏州_route_long | 苏州 | poi_search | False | True | False | None | True | True | False | False |  | 391.6 | 8 | 0 | 1.0000 | 0.0000 | True |  |  |
| 苏州_route_long | 苏州 | weather | False | True | False | None | True | True | False | False |  | 232.64 | None | 0 |  |  | True |  |  |
| 苏州_route_long | 苏州 | route | False | True | False | None | True | True | False | False | amap | 888.34 | None | 0 |  |  | True |  |  |
| 苏州_route_long | 苏州 | restaurant_search | False | True | False | None | True | True | False | False |  | 197.64 | 2 | 0 | 1.0000 | 0.0000 | True |  |  |
| 苏州_route_long | 苏州 | hotel_search | False | True | False | None | True | True | False | False |  | 285.09 | 10 | 0 | 1.0000 | 0.0000 | True |  |  |
| 苏州_route_long | 苏州 | budget_estimate | False | True | False | None | True | True | False | False |  | 0.73 | 1 | 0 |  |  | True |  |  |
| 武汉_food | 武汉 | poi_search | False | True | False | None | True | True | False | False |  | 459.85 | 20 | 0 | 0.9500 | 0.0000 | True |  |  |
| 武汉_food | 武汉 | weather | False | True | False | None | True | True | False | False |  | 217.77 | None | 0 |  |  | True |  |  |
| 武汉_food | 武汉 | route | False | True | False | None | True | True | False | False | amap | 742.59 | None | 0 |  |  | True |  |  |
| 武汉_food | 武汉 | restaurant_search | False | True | False | None | True | True | False | False |  | 291.28 | 10 | 0 | 0.8000 | 0.0000 | True |  |  |
| 武汉_food | 武汉 | hotel_search | False | True | False | None | True | True | False | False |  | 276.62 | 10 | 0 | 1.0000 | 0.0000 | True |  |  |
| 武汉_food | 武汉 | budget_estimate | False | True | False | None | True | True | False | False |  | 0.69 | 1 | 0 |  |  | True |  |  |
| 武汉_history | 武汉 | poi_search | False | True | False | None | True | True | False | False |  | 177.61 | 1 | 0 | 1.0000 | 0.0000 | True |  |  |
| 武汉_history | 武汉 | weather | False | True | False | None | True | True | False | False |  | 215.93 | None | 0 |  |  | True |  |  |
| 武汉_history | 武汉 | route | False | True | False | None | True | True | False | False | amap_fallback_estimate | 494.29 | None | 0 |  |  | True |  |  |
| 武汉_history | 武汉 | restaurant_search | False | True | True | primary_unusable | True | False | True | False |  | 79.32 | 0 | 0 |  |  | True |  |  |
| 武汉_history | 武汉 | hotel_search | False | True | False | None | True | True | False | False |  | 321.36 | 10 | 0 | 1.0000 | 0.0000 | True |  |  |
| 武汉_history | 武汉 | budget_estimate | False | True | False | None | True | True | False | False |  | 0.64 | 1 | 0 |  |  | True |  |  |
| 武汉_nature | 武汉 | poi_search | False | True | False | None | True | True | False | False |  | 542.62 | 20 | 0 | 1.0000 | 0.0000 | True |  |  |
| 武汉_nature | 武汉 | weather | False | True | False | None | True | True | False | False |  | 798.61 | None | 0 |  |  | True |  |  |
| 武汉_nature | 武汉 | route | False | True | False | None | True | True | False | False | amap | 918.41 | None | 0 |  |  | True |  |  |
| 武汉_nature | 武汉 | restaurant_search | False | True | False | None | True | True | False | False |  | 211.13 | 3 | 0 | 0.6667 | 0.0000 | True |  |  |
| 武汉_nature | 武汉 | hotel_search | False | True | False | None | True | True | False | False |  | 319.87 | 10 | 0 | 1.0000 | 0.0000 | True |  |  |
| 武汉_nature | 武汉 | budget_estimate | False | True | False | None | True | True | False | False |  | 0.75 | 1 | 0 |  |  | True |  |  |
| 武汉_shopping | 武汉 | poi_search | False | True | False | None | True | True | False | False |  | 447.69 | 20 | 0 | 0.9500 | 0.0000 | True |  |  |
| 武汉_shopping | 武汉 | weather | False | True | False | None | True | True | False | False |  | 212.72 | None | 0 |  |  | True |  |  |
| 武汉_shopping | 武汉 | route | False | True | False | None | True | True | False | False | amap | 650.38 | None | 0 |  |  | True |  |  |
| 武汉_shopping | 武汉 | restaurant_search | False | True | False | None | True | True | False | False |  | 209.29 | 3 | 0 | 0.6667 | 0.0000 | True |  |  |
| 武汉_shopping | 武汉 | hotel_search | False | True | False | None | True | True | False | False |  | 273.87 | 10 | 0 | 1.0000 | 0.0000 | True |  |  |
| 武汉_shopping | 武汉 | budget_estimate | False | True | False | None | True | True | False | False |  | 0.53 | 1 | 0 |  |  | True |  |  |
| 武汉_family | 武汉 | poi_search | False | True | True | primary_unusable | True | False | True | False |  | 97.6 | 0 | 0 |  |  | True |  |  |
| 武汉_family | 武汉 | weather | False | True | False | None | True | True | False | False |  | 251.8 | None | 0 |  |  | True |  |  |
| 武汉_family | 武汉 | route | False | True | False | None | True | True | False | False | amap_fallback_estimate | 476.49 | None | 0 |  |  | True |  |  |
| 武汉_family | 武汉 | restaurant_search | False | True | False | None | True | True | False | False |  | 244.17 | 3 | 0 | 0.6667 | 0.0000 | True |  |  |
| 武汉_family | 武汉 | hotel_search | False | True | True | primary_unusable | True | False | True | False |  | 99.14 | 0 | 0 |  |  | True |  |  |
| 武汉_family | 武汉 | budget_estimate | False | True | False | None | True | True | False | False |  | 0.45 | 1 | 0 |  |  | True |  |  |
| 武汉_museum | 武汉 | poi_search | False | True | False | None | True | True | False | False |  | 296.6 | 20 | 0 | 1.0000 | 0.0000 | True |  |  |
| 武汉_museum | 武汉 | weather | False | True | False | None | True | True | False | False |  | 234.84 | None | 0 |  |  | True |  |  |
| 武汉_museum | 武汉 | route | False | True | False | None | True | True | False | False | amap | 567.25 | None | 0 |  |  | True |  |  |
| 武汉_museum | 武汉 | restaurant_search | False | True | False | None | True | True | False | False |  | 214.41 | 3 | 0 | 0.6667 | 0.0000 | True |  |  |
| 武汉_museum | 武汉 | hotel_search | False | True | False | None | True | True | False | False |  | 285.88 | 10 | 0 | 1.0000 | 0.0000 | True |  |  |
| 武汉_museum | 武汉 | budget_estimate | False | True | False | None | True | True | False | False |  | 0.5 | 1 | 0 |  |  | True |  |  |
| 武汉_night_food | 武汉 | poi_search | False | True | False | None | True | True | False | False |  | 288.23 | 20 | 0 | 1.0000 | 0.0000 | True |  |  |
| 武汉_night_food | 武汉 | weather | False | True | False | None | True | True | False | False |  | 244.41 | None | 0 |  |  | True |  |  |
| 武汉_night_food | 武汉 | route | False | True | False | None | True | True | False | False | amap | 618.47 | None | 0 |  |  | True |  |  |
| 武汉_night_food | 武汉 | restaurant_search | False | True | False | None | True | True | False | False |  | 242.87 | 10 | 0 | 1.0000 | 0.0000 | True |  |  |
| 武汉_night_food | 武汉 | hotel_search | False | True | False | None | True | True | False | False |  | 281.64 | 10 | 0 | 1.0000 | 0.0000 | True |  |  |
| 武汉_night_food | 武汉 | budget_estimate | False | True | False | None | True | True | False | False |  | 1.25 | 1 | 0 |  |  | True |  |  |
| 武汉_route_long | 武汉 | poi_search | False | True | False | None | True | True | False | False |  | 270.52 | 9 | 0 | 1.0000 | 0.0000 | True |  |  |
| 武汉_route_long | 武汉 | weather | False | True | False | None | True | True | False | False |  | 204.88 | None | 0 |  |  | True |  |  |
| 武汉_route_long | 武汉 | route | False | True | False | None | True | True | False | False | amap | 819.44 | None | 0 |  |  | True |  |  |
| 武汉_route_long | 武汉 | restaurant_search | False | True | False | None | True | True | False | False |  | 224.76 | 3 | 0 | 0.6667 | 0.0000 | True |  |  |
| 武汉_route_long | 武汉 | hotel_search | False | True | False | None | True | True | False | False |  | 279.14 | 10 | 0 | 1.0000 | 0.0000 | True |  |  |
| 武汉_route_long | 武汉 | budget_estimate | False | True | False | None | True | True | False | False |  | 0.61 | 1 | 0 |  |  | True |  |  |
| missing_city_mars | 不存在的火星城 | poi_search | True | True | True | primary_unusable | True | False | True | False |  | 263.57 | 0 | 0 |  |  | True |  |  |
| missing_city_mars | 不存在的火星城 | weather | True | True | True | primary_unusable | True | False | False | False |  | 234.0 | None | 0 |  |  | True |  |  |
| missing_city_mars | 不存在的火星城 | route | True | True | True | primary_unusable | True | False | True | True |  | 717.34 | None | 0 |  |  | True |  |  |
| missing_city_mars | 不存在的火星城 | restaurant_search | True | True | True | primary_unusable | True | False | True | False |  | 294.73 | 0 | 0 |  |  | True |  |  |
| missing_city_mars | 不存在的火星城 | hotel_search | True | True | True | primary_unusable | True | False | True | False |  | 478.1 | 0 | 0 |  |  | True |  |  |
| missing_city_mars | 不存在的火星城 | budget_estimate | True | True | False | None | True | True | False | False |  | 0.69 | 1 | 0 |  |  | True |  |  |
| ambiguous_city_south | 南方 | poi_search | True | True | True | primary_unusable | True | False | True | False |  | 724.78 | 0 | 0 |  |  | True |  |  |
| ambiguous_city_south | 南方 | weather | True | True | True | primary_unusable | True | False | False | False |  | 247.97 | None | 0 |  |  | True |  |  |
| ambiguous_city_south | 南方 | route | True | True | True | primary_unusable | True | False | True | True |  | 600.82 | None | 0 |  |  | True |  |  |
| ambiguous_city_south | 南方 | restaurant_search | True | True | True | primary_unusable | True | False | True | False |  | 245.33 | 0 | 0 |  |  | True |  |  |
| ambiguous_city_south | 南方 | hotel_search | True | True | True | primary_unusable | True | False | True | False |  | 98.93 | 0 | 0 |  |  | True |  |  |
| ambiguous_city_south | 南方 | budget_estimate | True | True | False | None | True | True | False | False |  | 0.63 | 1 | 0 |  |  | True |  |  |
| ambiguous_city_oldtown | 古城 | poi_search | True | True | True | primary_unusable | True | False | True | False |  | 192.57 | 0 | 0 |  |  | True |  |  |
| ambiguous_city_oldtown | 古城 | weather | True | True | True | primary_unusable | True | False | False | False |  | 256.58 | None | 0 |  |  | True |  |  |
| ambiguous_city_oldtown | 古城 | route | True | True | True | primary_unusable | True | False | True | True |  | 526.09 | None | 0 |  |  | True |  |  |
| ambiguous_city_oldtown | 古城 | restaurant_search | True | True | True | primary_unusable | True | False | True | False |  | 192.42 | 0 | 0 |  |  | True |  |  |
| ambiguous_city_oldtown | 古城 | hotel_search | True | True | True | primary_unusable | True | False | True | False |  | 237.72 | 0 | 0 |  |  | True |  |  |
| ambiguous_city_oldtown | 古城 | budget_estimate | True | True | False | None | True | True | False | False |  | 1.17 | 1 | 0 |  |  | True |  |  |
| narrow_keyword_quantum | 杭州 | poi_search | True | True | True | primary_unusable | True | False | True | False |  | 339.77 | 0 | 0 |  |  | True |  |  |
| narrow_keyword_quantum | 杭州 | weather | True | True | True | primary_unusable | True | False | False | False |  | 243.02 | None | 0 |  |  | True |  |  |
| narrow_keyword_quantum | 杭州 | route | True | True | True | primary_unusable | True | False | True | True |  | 935.74 | None | 0 |  |  | True |  |  |
| narrow_keyword_quantum | 杭州 | restaurant_search | True | True | True | primary_unusable | True | False | False | False |  | 232.66 | 2 | 0 | 1.0000 | 0.0000 | True |  |  |
| narrow_keyword_quantum | 杭州 | hotel_search | True | True | True | primary_unusable | True | False | True | False |  | 333.05 | 0 | 0 |  |  | True |  |  |
| narrow_keyword_quantum | 杭州 | budget_estimate | True | True | False | None | True | True | False | False |  | 0.67 | 1 | 0 |  |  | True |  |  |
| narrow_keyword_space | 北京 | poi_search | True | True | True | primary_unusable | True | False | True | False |  | 327.56 | 0 | 0 |  |  | True |  |  |
| narrow_keyword_space | 北京 | weather | True | True | True | primary_unusable | True | False | False | False |  | 236.08 | None | 0 |  |  | True |  |  |
| narrow_keyword_space | 北京 | route | True | True | True | primary_unusable | True | False | True | True |  | 605.94 | None | 0 |  |  | True |  |  |
| narrow_keyword_space | 北京 | restaurant_search | True | True | True | primary_unusable | True | False | False | False |  | 328.21 | 2 | 0 | 1.0000 | 0.0000 | True |  |  |
| narrow_keyword_space | 北京 | hotel_search | True | True | True | primary_unusable | True | False | True | False |  | 309.88 | 0 | 0 |  |  | True |  |  |
| narrow_keyword_space | 北京 | budget_estimate | True | True | False | None | True | True | False | False |  | 0.83 | 1 | 0 |  |  | True |  |  |
| wrong_category_food_in_scenic | 上海 | poi_search | False | True | False | None | True | True | False | False |  | 381.78 | 20 | 0 | 0.8500 | 0.0000 | True |  |  |
| wrong_category_food_in_scenic | 上海 | weather | False | True | False | None | True | True | False | False |  | 228.96 | None | 0 |  |  | True |  |  |
| wrong_category_food_in_scenic | 上海 | route | False | True | False | None | True | True | False | False | amap | 758.75 | None | 0 |  |  | True |  |  |
| wrong_category_food_in_scenic | 上海 | restaurant_search | False | True | False | None | True | True | False | False |  | 351.6 | 10 | 0 | 0.9000 | 0.0000 | True |  |  |
| wrong_category_food_in_scenic | 上海 | hotel_search | False | True | False | None | True | True | False | False |  | 291.91 | 10 | 0 | 1.0000 | 0.0000 | True |  |  |
| wrong_category_food_in_scenic | 上海 | budget_estimate | False | True | False | None | True | True | False | False |  | 0.79 | 1 | 0 |  |  | True |  |  |
| wrong_category_museum_in_food | 成都 | poi_search | False | True | False | None | True | True | False | False |  | 257.56 | 20 | 0 | 0.8500 | 0.0000 | True |  |  |
| wrong_category_museum_in_food | 成都 | weather | False | True | False | None | True | True | False | False |  | 231.9 | None | 0 |  |  | True |  |  |
| wrong_category_museum_in_food | 成都 | route | False | True | False | None | True | True | False | False | amap | 523.61 | None | 0 |  |  | True |  |  |
| wrong_category_museum_in_food | 成都 | restaurant_search | False | True | False | None | True | True | False | False |  | 216.89 | 2 | 0 | 1.0000 | 0.0000 | True |  |  |
| wrong_category_museum_in_food | 成都 | hotel_search | False | True | False | None | True | True | False | False |  | 293.73 | 10 | 0 | 1.0000 | 0.0000 | True |  |  |
| wrong_category_museum_in_food | 成都 | budget_estimate | False | True | False | None | True | True | False | False |  | 0.75 | 1 | 0 |  |  | True |  |  |
| route_sparse_city | 苏州 | poi_search | False | True | True | primary_unusable | True | False | True | False |  | 95.72 | 0 | 0 |  |  | True |  |  |
| route_sparse_city | 苏州 | weather | False | True | False | None | True | True | False | False |  | 242.29 | None | 0 |  |  | True |  |  |
| route_sparse_city | 苏州 | route | False | True | False | None | True | True | False | False | amap | 505.95 | None | 0 |  |  | True |  |  |
| route_sparse_city | 苏州 | restaurant_search | False | True | False | None | True | True | False | False |  | 222.42 | 2 | 0 | 1.0000 | 0.0000 | True |  |  |
| route_sparse_city | 苏州 | hotel_search | False | True | False | None | True | True | False | False |  | 614.83 | 10 | 0 | 1.0000 | 0.0000 | True |  |  |
| route_sparse_city | 苏州 | budget_estimate | False | True | False | None | True | True | False | False |  | 0.5 | 1 | 0 |  |  | True |  |  |
| route_cross_district | 重庆 | poi_search | False | True | False | None | True | True | False | False |  | 553.88 | 20 | 0 | 1.0000 | 0.0000 | True |  |  |
| route_cross_district | 重庆 | weather | False | True | False | None | True | True | False | False |  | 221.16 | None | 0 |  |  | True |  |  |
| route_cross_district | 重庆 | route | False | True | False | None | True | True | False | False | amap_fallback_estimate | 570.97 | None | 0 |  |  | True |  |  |
| route_cross_district | 重庆 | restaurant_search | False | True | False | None | True | True | False | False |  | 221.04 | 1 | 0 | 1.0000 | 0.0000 | True |  |  |
| route_cross_district | 重庆 | hotel_search | False | True | False | None | True | True | False | False |  | 272.69 | 10 | 0 | 1.0000 | 0.0000 | True |  |  |
| route_cross_district | 重庆 | budget_estimate | False | True | False | None | True | True | False | False |  | 0.42 | 1 | 0 |  |  | True |  |  |
| route_airport_like | 深圳 | poi_search | False | True | False | None | True | True | False | False |  | 309.8 | 20 | 0 | 1.0000 | 0.0000 | True |  |  |
| route_airport_like | 深圳 | weather | False | True | False | None | True | True | False | False |  | 233.17 | None | 0 |  |  | True |  |  |
| route_airport_like | 深圳 | route | False | True | False | None | True | True | False | False | amap | 505.09 | None | 0 |  |  | True |  |  |
| route_airport_like | 深圳 | restaurant_search | False | True | False | None | True | True | False | False |  | 239.92 | 6 | 0 | 1.0000 | 0.0000 | True |  |  |
| route_airport_like | 深圳 | hotel_search | False | True | False | None | True | True | False | False |  | 324.35 | 10 | 0 | 1.0000 | 0.0000 | True |  |  |
| route_airport_like | 深圳 | budget_estimate | False | True | False | None | True | True | False | False |  | 0.45 | 1 | 0 |  |  | True |  |  |
| weather_invalid_city | 不存在天气城 | poi_search | True | True | True | primary_unusable | True | False | True | False |  | 344.72 | 0 | 0 |  |  | True |  |  |
| weather_invalid_city | 不存在天气城 | weather | True | True | True | primary_unusable | True | False | False | False |  | 228.36 | None | 0 |  |  | True |  |  |
| weather_invalid_city | 不存在天气城 | route | True | True | True | primary_unusable | True | False | True | True |  | 974.27 | None | 0 |  |  | True |  |  |
| weather_invalid_city | 不存在天气城 | restaurant_search | True | True | True | primary_unusable | True | False | True | False |  | 248.03 | 0 | 0 |  |  | True |  |  |
| weather_invalid_city | 不存在天气城 | hotel_search | True | True | True | primary_unusable | True | False | True | False |  | 342.58 | 0 | 0 |  |  | True |  |  |
| weather_invalid_city | 不存在天气城 | budget_estimate | True | True | False | None | True | True | False | False |  | 0.77 | 1 | 0 |  |  | True |  |  |
| duplicate_risk_common_food | 广州 | poi_search | False | True | False | None | True | True | False | False |  | 380.57 | 20 | 0 | 1.0000 | 0.0000 | True |  |  |
| duplicate_risk_common_food | 广州 | weather | False | True | False | None | True | True | False | False |  | 225.03 | None | 0 |  |  | True |  |  |
| duplicate_risk_common_food | 广州 | route | False | True | False | None | True | True | False | False | amap | 669.63 | None | 0 |  |  | True |  |  |
| duplicate_risk_common_food | 广州 | restaurant_search | False | True | False | None | True | True | False | False |  | 350.71 | 10 | 0 | 0.9000 | 0.0000 | True |  |  |
| duplicate_risk_common_food | 广州 | hotel_search | False | True | False | None | True | True | False | False |  | 249.13 | 10 | 0 | 1.0000 | 0.0000 | True |  |  |
| duplicate_risk_common_food | 广州 | budget_estimate | False | True | False | None | True | True | False | False |  | 0.73 | 1 | 0 |  |  | True |  |  |
| duplicate_risk_common_scenic | 北京 | poi_search | False | True | False | None | True | True | False | False |  | 298.19 | 20 | 0 | 1.0000 | 0.0000 | True |  |  |
| duplicate_risk_common_scenic | 北京 | weather | False | True | False | None | True | True | False | False |  | 235.63 | None | 0 |  |  | True |  |  |
| duplicate_risk_common_scenic | 北京 | route | False | True | False | None | True | True | False | False | amap | 533.45 | None | 0 |  |  | True |  |  |
| duplicate_risk_common_scenic | 北京 | restaurant_search | False | True | False | None | True | True | False | False |  | 294.58 | 10 | 0 | 1.0000 | 0.0000 | True |  |  |
| duplicate_risk_common_scenic | 北京 | hotel_search | False | True | False | None | True | True | False | False |  | 324.29 | 10 | 0 | 1.0000 | 0.0000 | True |  |  |
| duplicate_risk_common_scenic | 北京 | budget_estimate | False | True | False | None | True | True | False | False |  | 1.25 | 1 | 0 |  |  | True |  |  |
| empty_query_symbols | @@@ | poi_search | True | True | True | primary_unusable | True | False | True | False |  | 109.89 | 0 | 0 |  |  | True |  |  |
| empty_query_symbols | @@@ | weather | True | True | True | primary_unusable | True | False | False | False |  | 218.53 | None | 0 |  |  | True |  |  |
| empty_query_symbols | @@@ | route | True | True | True | primary_unusable | True | False | True | True |  | 252.73 | None | 0 |  |  | True |  |  |
| empty_query_symbols | @@@ | restaurant_search | True | True | True | primary_unusable | True | False | True | False |  | 258.96 | 0 | 0 |  |  | True |  |  |
| empty_query_symbols | @@@ | hotel_search | True | True | True | primary_unusable | True | False | True | False |  | 109.4 | 0 | 0 |  |  | True |  |  |
| empty_query_symbols | @@@ | budget_estimate | True | True | False | None | True | True | False | False |  | 1.24 | 1 | 0 |  |  | True |  |  |
| overbroad_keyword | 武汉 | poi_search | False | True | True | primary_unusable | True | False | True | False |  | 109.48 | 0 | 0 |  |  | True |  |  |
| overbroad_keyword | 武汉 | weather | False | True | False | None | True | True | False | False |  | 220.9 | None | 0 |  |  | True |  |  |
| overbroad_keyword | 武汉 | route | False | True | False | None | True | True | False | False | amap | 605.6 | None | 0 |  |  | True |  |  |
| overbroad_keyword | 武汉 | restaurant_search | False | True | False | None | True | True | False | False |  | 245.88 | 3 | 0 | 0.6667 | 0.0000 | True |  |  |
| overbroad_keyword | 武汉 | hotel_search | False | True | True | primary_unusable | True | False | True | False |  | 96.43 | 0 | 0 |  |  | True |  |  |
| overbroad_keyword | 武汉 | budget_estimate | False | True | False | None | True | True | False | False |  | 0.52 | 1 | 0 |  |  | True |  |  |
| english_city_alias | Shanghai | poi_search | False | True | False | None | True | True | False | False |  | 542.94 | 20 | 0 | 1.0000 | 0.0000 | True |  |  |
| english_city_alias | Shanghai | weather | False | True | False | None | True | True | False | False |  | 760.45 | None | 0 |  |  | True |  |  |
| english_city_alias | Shanghai | route | False | True | False | None | True | True | False | False | amap | 659.4 | None | 0 |  |  | True |  |  |
| english_city_alias | Shanghai | restaurant_search | False | True | False | None | True | True | False | False |  | 232.4 | 1 | 0 | 1.0000 | 0.0000 | True |  |  |
| english_city_alias | Shanghai | hotel_search | False | True | False | None | True | True | False | False |  | 329.18 | 10 | 0 | 1.0000 | 0.0000 | True |  |  |
| english_city_alias | Shanghai | budget_estimate | False | True | False | None | True | True | False | False |  | 0.49 | 1 | 0 |  |  | True |  |  |
| mixed_language_city | Hangzhou | poi_search | False | True | False | None | True | True | False | False |  | 188.12 | 1 | 0 | 1.0000 | 0.0000 | True |  |  |
| mixed_language_city | Hangzhou | weather | False | True | False | None | True | True | False | False |  | 245.1 | None | 0 |  |  | True |  |  |
| mixed_language_city | Hangzhou | route | False | True | False | None | True | True | False | False | amap | 770.04 | None | 0 |  |  | True |  |  |
| mixed_language_city | Hangzhou | restaurant_search | False | True | False | None | True | True | False | False |  | 237.85 | 5 | 0 | 1.0000 | 0.0000 | True |  |  |
| mixed_language_city | Hangzhou | hotel_search | False | True | False | None | True | True | False | False |  | 370.2 | 10 | 0 | 1.0000 | 0.0000 | True |  |  |
| mixed_language_city | Hangzhou | budget_estimate | False | True | False | None | True | True | False | False |  | 0.66 | 1 | 0 |  |  | True |  |  |
| district_only | 朝阳区 | poi_search | True | True | True | primary_unusable | True | False | True | False |  | 397.03 | 0 | 0 |  |  | True |  |  |
| district_only | 朝阳区 | weather | True | True | True | primary_unusable | True | False | False | False |  | 259.74 | None | 0 |  |  | True |  |  |
| district_only | 朝阳区 | route | True | True | True | primary_unusable | True | False | True | True |  | 620.9 | None | 0 |  |  | True |  |  |
| district_only | 朝阳区 | restaurant_search | True | True | True | primary_unusable | True | False | True | False |  | 336.93 | 0 | 0 |  |  | True |  |  |
| district_only | 朝阳区 | hotel_search | True | True | True | primary_unusable | True | False | True | False |  | 108.61 | 0 | 0 |  |  | True |  |  |
| district_only | 朝阳区 | budget_estimate | True | True | False | None | True | True | False | False |  | 0.76 | 1 | 0 |  |  | True |  |  |
| landmark_as_city | 西湖 | poi_search | True | True | True | primary_unusable | True | False | True | False |  | 270.0 | 0 | 0 |  |  | True |  |  |
| landmark_as_city | 西湖 | weather | True | True | True | primary_unusable | True | False | False | False |  | 209.32 | None | 0 |  |  | True |  |  |
| landmark_as_city | 西湖 | route | True | True | True | primary_unusable | True | False | True | True |  | 693.2 | None | 0 |  |  | True |  |  |
| landmark_as_city | 西湖 | restaurant_search | True | True | True | primary_unusable | True | False | True | False |  | 210.45 | 0 | 0 |  |  | True |  |  |
| landmark_as_city | 西湖 | hotel_search | True | True | True | primary_unusable | True | False | True | False |  | 220.99 | 0 | 0 |  |  | True |  |  |
| landmark_as_city | 西湖 | budget_estimate | True | True | False | None | True | True | False | False |  | 0.41 | 1 | 0 |  |  | True |  |  |
| hotel_area_like | 春熙路 | poi_search | True | True | True | primary_unusable | True | False | True | False |  | 318.75 | 0 | 0 |  |  | True |  |  |
| hotel_area_like | 春熙路 | weather | True | True | True | primary_unusable | True | False | False | False |  | 209.03 | None | 0 |  |  | True |  |  |
| hotel_area_like | 春熙路 | route | True | True | True | primary_unusable | True | False | True | True |  | 535.74 | None | 0 |  |  | True |  |  |
| hotel_area_like | 春熙路 | restaurant_search | True | True | True | primary_unusable | True | False | True | False |  | 216.32 | 0 | 0 |  |  | True |  |  |
| hotel_area_like | 春熙路 | hotel_search | True | True | True | primary_unusable | True | False | True | False |  | 91.16 | 0 | 0 |  |  | True |  |  |
| hotel_area_like | 春熙路 | budget_estimate | True | True | False | None | True | True | False | False |  | 0.47 | 1 | 0 |  |  | True |  |  |

## 说明

- `live` 模式会调用当前配置的 provider，并保存每个 case 的 API/snapshot payload；
- `replay` 模式只读取 snapshot，不再调用外部 API，用于复现报告；
- 如果未配置高德 key，`provider configured=False`，结果主要反映本地 fallback 链路；
- `chinatravel` provider 使用 ChinaTravel sandbox/database 或 synthetic ChinaTravel provider，不调用外部 API；
- 除 POI / weather / route 外，灰度测试还覆盖 restaurant_search / hotel_search / budget_estimate；
- `primary_ok=True` 表示目标 provider 调用未抛异常；`primary_usable=True` 表示其结果可直接用于 agent；
- `fallback=True` 表示该操作最终使用了兜底结果，例如本地 POI、mock weather 或 haversine route。