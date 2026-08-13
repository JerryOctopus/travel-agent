"""Build the committed long_horizon_v1 supplementary evaluation set."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "eval" / "production_v1"
DEV_CASE_IDS = {"lh_5_001", "lh_8_001", "lh_12_001", "lh_12_002"}
CORE_FROZEN_CASE_IDS = {"lh_5_002", "lh_5_003", "lh_5_004", "lh_8_002"}


def _split(case_id: str) -> str:
    if case_id in DEV_CASE_IDS:
        return "dev"
    if case_id in CORE_FROZEN_CASE_IDS:
        return "core_frozen"
    return "challenge_frozen"


def _case(
    case_id: str,
    title: str,
    steps: list[tuple[str, dict[str, Any]]],
) -> dict[str, Any]:
    state: dict[str, Any] = {}
    conversation: list[dict[str, Any]] = []
    expectations: list[dict[str, Any]] = []
    for turn, (content, update) in enumerate(steps, 1):
        state.update(copy.deepcopy(update))
        conversation.append({"turn": turn, "role": "user", "content": content})
        expectations.append({"turn": turn, "constraints": copy.deepcopy(state)})
    return {
        "case_id": case_id,
        "dataset_version": "travel-agent-eval-production-v1.1",
        "schema_version": "1.1",
        "split": _split(case_id),
        "subset": "long_horizon_state",
        "title": title,
        "reference_datetime": "2026-08-11T12:00:00+08:00",
        "difficulty": "hard" if len(steps) >= 12 else "medium",
        "turn_mode": "multi_turn",
        "conversation": conversation,
        "user_query": conversation[0]["content"],
        "turn_expectations": expectations,
        "tags": ["multi_turn", "long_horizon", f"turns_{len(steps)}"],
        "gold": {
            "constraints": state,
            "must_clarify_before_plan": [],
            "expected_outcome": "full_plan",
            "required_behaviors": [
                "逐轮维护约束状态",
                "后出现的明确修改覆盖旧值",
                "未被修改的早期约束必须保留",
            ],
            "forbidden_behaviors": [
                "恢复已明确删除的约束",
                "用后续软偏好覆盖硬约束",
            ],
        },
        "required_tools": ["poi_search", "route_planning", "plan_and_critique"],
        "fixture_id": f"fixture_{case_id}_v1",
        "fixture_profile": "normal",
        "evaluators": [
            "turn_state_retention",
            "constraint_extraction",
            "constraint_satisfaction",
            "tool_trace",
        ],
        "source_style": "curated",
        "architecture_policy": {
            "shared_tool": "plan_and_critique",
            "shared_tool_mode": "deterministic_closed_loop",
            "shared_across_versions": ["V0", "V1", "V2", "V3"],
            "external_semantic_critic_only_in": ["V3"],
        },
    }


CASES = [
    _case("lh_5_001", "预算与老人约束逐步补充", [
        ("10月去杭州玩三天。", {"destinations": ["杭州"], "duration_days": 3}),
        ("我们两个人，其中一位是72岁老人。", {"traveler_count": 2, "elderly": True}),
        ("总预算最多3000元。", {"budget_max_cny": 3000}),
        ("住西湖东侧，每天步行不超过6公里。", {"lodging_area": "西湖东侧", "max_walking_km_per_day": 6}),
        ("最后方案里必须保留西湖，预算不要增加。", {"must_visit": ["西湖"]}),
    ]),
    _case("lh_5_002", "景点替换后保留其余条件", [
        ("周末去苏州两天，想去拙政园和虎丘。", {"destinations": ["苏州"], "duration_days": 2, "must_visit": ["拙政园", "虎丘"]}),
        ("住观前街附近。", {"lodging_area": "观前街附近"}),
        ("两个人，总预算2500元。", {"traveler_count": 2, "budget_max_cny": 2500}),
        ("虎丘不去了，换成苏州博物馆。", {"must_visit": ["拙政园", "苏州博物馆"], "removed": ["虎丘"]}),
        ("节奏轻松一点，其他要求照旧。", {"pace": "relaxed"}),
    ]),
    _case("lh_5_003", "固定活动与返程时间", [
        ("9月14日去西安三天，住钟楼附近。", {"destinations": ["西安"], "duration_days": 3, "date_start": "2026-09-14", "lodging_area": "钟楼附近"}),
        ("两个人，想看历史景点。", {"traveler_count": 2, "interests": ["历史景点"]}),
        ("第二天下午14:30到16:30预约了陕西历史博物馆。", {"fixed_events": [{"day": 2, "start": "14:30", "end": "16:30", "location": "陕西历史博物馆"}]}),
        ("第三天18点前必须到西安北站。", {"return_deadline": "2026-09-16T18:00:00+08:00"}),
        ("预算上限4000元，前面的预约和返程都不能动。", {"budget_max_cny": 4000}),
    ]),
    _case("lh_5_004", "交通方式变更", [
        ("青岛玩三天，住市南区，本来准备自驾。", {"destinations": ["青岛"], "duration_days": 3, "lodging_area": "市南区", "self_driving_allowed": True}),
        ("一家三口，孩子6岁。", {"traveler_count": 3, "child_age": 6}),
        ("总预算4500元。", {"budget_max_cny": 4500}),
        ("驾照没带，取消自驾，只坐公交地铁或打车。", {"self_driving_allowed": False, "transport_modes": ["public_transport", "taxi"]}),
        ("崂山可以删掉，但住宿区域和预算不变。", {"optional_remove": ["崂山"]}),
    ]),
    _case("lh_8_001", "多次预算调整与必去保留", [
        ("两个人去厦门四天。", {"destinations": ["厦门"], "duration_days": 4, "traveler_count": 2}),
        ("最初预算6000元。", {"budget_max_cny": 6000}),
        ("鼓浪屿必须去，也想看看海边。", {"must_visit": ["鼓浪屿"], "interests": ["海边"]}),
        ("住宿放在思明区。", {"lodging_area": "思明区"}),
        ("其中一人不吃海鲜。", {"dietary": ["不吃海鲜"]}),
        ("预算降到4800元，住宿可以降档。", {"budget_max_cny": 4800, "lodging_flexibility": "can_downgrade"}),
        ("行程节奏改成轻松。", {"pace": "relaxed"}),
        ("再确认一次：鼓浪屿仍是必去，不吃海鲜也保留。", {}),
    ]),
    _case("lh_8_002", "会议插入与日期保持", [
        ("11月2日到成都，安排四天。", {"destinations": ["成都"], "duration_days": 4, "date_start": "2026-11-02"}),
        ("三个人，预算7000元。", {"traveler_count": 3, "budget_max_cny": 7000}),
        ("住春熙路附近。", {"lodging_area": "春熙路附近"}),
        ("想去熊猫基地和武侯祠。", {"must_visit": ["熊猫基地", "武侯祠"]}),
        ("第三天上午9点到11点有线上会议。", {"fixed_events": [{"day": 3, "start": "09:00", "end": "11:00", "location": "酒店"}]}),
        ("会议需要提前半小时回酒店。", {"buffer_before_fixed_event_min": 30}),
        ("不要安排太辣的餐厅。", {"dietary": ["少辣"]}),
        ("熊猫基地别删，其他按前面条件调整。", {}),
    ]),
    _case("lh_8_003", "人数变化与房型更新", [
        ("南京三日游，先按两个人计划。", {"destinations": ["南京"], "duration_days": 3, "traveler_count": 2}),
        ("住新街口附近。", {"lodging_area": "新街口附近"}),
        ("预算4000元。", {"budget_max_cny": 4000}),
        ("中山陵必须去。", {"must_visit": ["中山陵"]}),
        ("后来又有两位朋友加入，一共四人。", {"traveler_count": 4}),
        ("酒店改成两间双床房。", {"room_requirement": "2 twin rooms"}),
        ("预算可以提高到6500元。", {"budget_max_cny": 6500}),
        ("住宿位置仍然是新街口，中山陵也保留。", {}),
    ]),
    _case("lh_8_004", "取消项目后明确禁入", [
        ("北京玩五天，两个人。", {"destinations": ["北京"], "duration_days": 5, "traveler_count": 2}),
        ("预算8000元，住前门附近。", {"budget_max_cny": 8000, "lodging_area": "前门附近"}),
        ("故宫和长城都想去。", {"must_visit": ["故宫", "长城"]}),
        ("每天最多步行8公里。", {"max_walking_km_per_day": 8}),
        ("膝盖不舒服，取消长城。", {"must_visit": ["故宫"], "removed": ["长城"]}),
        ("改加国家博物馆。", {"must_visit": ["故宫", "国家博物馆"]}),
        ("节奏改为轻松。", {"pace": "relaxed"}),
        ("最后别因为经典路线又把长城加回来。", {}),
    ]),
    _case("lh_12_001", "打岔后的早期预算召回", [
        ("杭州五天，两个人，总预算5000元。", {"destinations": ["杭州"], "duration_days": 5, "traveler_count": 2, "budget_max_cny": 5000}),
        ("住西湖东侧。", {"lodging_area": "西湖东侧"}),
        ("西湖和良渚博物院必须去。", {"must_visit": ["西湖", "良渚博物院"]}),
        ("其中一位老人，每天步行最多6公里。", {"elderly": True, "max_walking_km_per_day": 6}),
        ("顺便问一句，杭州十月通常需要带外套吗？", {}),
        ("回到行程，饮食要清淡。", {"dietary": ["清淡"]}),
        ("第三天下午安排自由活动。", {"fixed_events": [{"day": 3, "start": "14:00", "end": "18:00", "location": "自由活动"}]}),
        ("良渚不去了，换成浙江省博物馆。", {"must_visit": ["西湖", "浙江省博物馆"], "removed": ["良渚博物院"]}),
        ("预算降到4500元。", {"budget_max_cny": 4500}),
        ("住宿区域不要改。", {}),
        ("最后一天17点前到杭州东站。", {"return_deadline": "2026-10-05T17:00:00+08:00"}),
        ("按全部条件出最终版，别恢复良渚。", {}),
    ]),
    _case("lh_12_002", "多次撤销与重新指定", [
        ("上海四天，三个人。", {"destinations": ["上海"], "duration_days": 4, "traveler_count": 3}),
        ("预算6000元，住人民广场附近。", {"budget_max_cny": 6000, "lodging_area": "人民广场附近"}),
        ("外滩和迪士尼必须去。", {"must_visit": ["外滩", "迪士尼"]}),
        ("有个8岁孩子，节奏别太赶。", {"child_age": 8, "pace": "relaxed"}),
        ("不去迪士尼了，排队太久。", {"must_visit": ["外滩"], "removed": ["迪士尼"]}),
        ("改去上海自然博物馆。", {"must_visit": ["外滩", "上海自然博物馆"]}),
        ("第二天晚饭已经约在陆家嘴。", {"fixed_events": [{"day": 2, "start": "18:00", "end": "20:00", "location": "陆家嘴"}]}),
        ("预算最多再加500，改成6500元。", {"budget_max_cny": 6500}),
        ("住宿仍住人民广场，不要跟着晚饭换酒店。", {}),
        ("有人提议迪士尼，如果时间紧就算了。", {}),
        ("明确一下，迪士尼仍然不去。", {}),
        ("生成最终计划，保留外滩、自然博物馆和晚饭预约。", {}),
    ]),
    _case("lh_12_003", "交通反复修改与硬截止", [
        ("广州到珠海玩四天，两个人。", {"destinations": ["广州", "珠海"], "duration_days": 4, "traveler_count": 2}),
        ("总预算5500元。", {"budget_max_cny": 5500}),
        ("前两晚住广州珠江新城。", {"lodging_area": "广州珠江新城"}),
        ("原计划全程租车。", {"self_driving_allowed": True}),
        ("广州塔必须去。", {"must_visit": ["广州塔"]}),
        ("驾照丢了，取消租车。", {"self_driving_allowed": False}),
        ("城内用公共交通，跨城坐高铁。", {"transport_modes": ["public_transport", "rail"]}),
        ("第三天中午前到珠海。", {"arrival_deadline": "day3 12:00"}),
        ("第四天18点前必须到珠海机场。", {"return_deadline": "day4 18:00 珠海机场"}),
        ("顺路的话能不能看看长隆？先不要设成必去。", {"optional_visit": ["长隆"]}),
        ("时间不够就删长隆，广州塔不能删。", {"optional_remove": ["长隆"]}),
        ("最终版严格按无自驾和两个截止时间安排。", {}),
    ]),
    _case("lh_12_004", "长对话中的日期与饮食保持", [
        ("2026年12月20日去哈尔滨五天，四个人。", {"destinations": ["哈尔滨"], "duration_days": 5, "date_start": "2026-12-20", "traveler_count": 4}),
        ("总预算9000元。", {"budget_max_cny": 9000}),
        ("住中央大街附近。", {"lodging_area": "中央大街附近"}),
        ("冰雪大世界必须去。", {"must_visit": ["冰雪大世界"]}),
        ("同行有人不吃猪肉。", {"dietary": ["不吃猪肉"]}),
        ("还带一个5岁孩子。", {"child_age": 5}),
        ("每天户外连续时间不要超过两小时。", {"max_continuous_outdoor_hours": 2}),
        ("第二天晚上已有聚餐。", {"fixed_events": [{"day": 2, "start": "18:00", "end": "20:00", "location": "中央大街"}]}),
        ("预算改成8500元。", {"budget_max_cny": 8500}),
        ("住宿不能换到别的区域。", {}),
        ("如果太冷可以少排一个普通景点，但冰雪大世界保留。", {"weather_flexibility": "drop_non_must_visit"}),
        ("请汇总最终方案，日期、饮食和孩子约束都别漏。", {}),
    ]),
]


def main() -> None:
    split_names = ("dev", "core_frozen", "challenge_frozen", "shadow_frozen")
    long_ids = {case["case_id"] for case in CASES}
    merged: dict[str, list[dict[str, Any]]] = {}
    for split in split_names:
        path = DATA_DIR / f"{split}.jsonl"
        existing = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        existing = [case for case in existing if case.get("case_id") not in long_ids]
        for case in existing:
            case["dataset_version"] = "travel-agent-eval-production-v1.1"
        merged[split] = existing + [case for case in CASES if case["split"] == split]
        payload = "\n".join(
            json.dumps(case, ensure_ascii=False, separators=(",", ":"))
            for case in merged[split]
        )
        path.write_text(payload + "\n", encoding="utf-8")

    all_cases = [case for split in split_names for case in merged[split]]
    all_payload = "\n".join(
        json.dumps(case, ensure_ascii=False, separators=(",", ":"))
        for case in all_cases
    )
    (DATA_DIR / "all_cases.jsonl").write_text(all_payload + "\n", encoding="utf-8")
    print(f"merged {len(CASES)} long-horizon cases into {len(all_cases)} production_v1 cases")


if __name__ == "__main__":
    main()
