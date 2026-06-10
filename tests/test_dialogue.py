from travel_agent.dialogue import DialogueSession


def test_dialogue_session_merges_profile_across_turns() -> None:
    session = DialogueSession()

    first_reply = session.send("我想轻松一点，喜欢自然和美食")
    second_reply = session.send("杭州三天")

    assert first_reply == "你想去哪个城市，计划玩几天？"
    assert "# 杭州3天轻松行程草案" in second_reply
    assert session.profile is not None
    assert session.profile.destination == "杭州"
    assert session.profile.days == 3
    assert session.profile.interests == ["nature", "food"]
    assert len(session.messages) == 4
    assert len(session.turns) == 2


def test_dialogue_session_reset_clears_state() -> None:
    session = DialogueSession()
    session.send("我想轻松一点，喜欢自然和美食")

    session.reset()

    assert session.profile is None
    assert session.last_result is None
    assert session.messages == []
    assert session.turns == []
