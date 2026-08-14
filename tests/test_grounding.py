import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grounding.validator import (
    WebShopGroundingValidator,
    extract_clickables_from_observation,
)


def test_valid_search():
    v = WebShopGroundingValidator()
    check = v.check("search[l'eau d'issey]", has_search_bar=True)
    assert check.parseable
    assert check.element_exists is True
    assert check.valid
    assert check.score >= 0.9


def test_search_without_bar():
    v = WebShopGroundingValidator()
    check = v.check("search[foo]", has_search_bar=False)
    assert check.element_exists is False
    assert not check.valid
    assert check.score <= 0.35


def test_click_nonexistent_element():
    v = WebShopGroundingValidator()
    check = v.check("click[Z999NONEXISTENT]", clickables=["buy now", "next >"])
    assert check.element_exists is False
    assert not check.valid
    assert check.score <= 0.35


def test_click_existing_element():
    v = WebShopGroundingValidator()
    check = v.check("click[Buy Now]", clickables=["buy now", "next >"])
    assert check.element_exists is True
    assert check.valid


def test_disabled_button():
    v = WebShopGroundingValidator()
    check = v.check(
        "click[Buy Now]",
        clickables=["buy now"],
        disabled=["buy now"],
    )
    assert check.button_clickable is False
    assert not check.valid
    assert check.score <= 0.55


def test_unparseable_action():
    v = WebShopGroundingValidator()
    check = v.check("press the red button")
    assert not check.parseable
    assert not check.valid
    assert check.score == 0.0


def test_page_loading():
    v = WebShopGroundingValidator()
    check = v.check("click[Buy Now]", clickables=["buy now"], observation="Loading... please wait")
    assert not check.page_loaded
    assert not check.valid


def test_state_changed():
    v = WebShopGroundingValidator()
    check = v.check(
        "click[Buy Now]",
        clickables=["buy now"],
        observation="page a",
        prev_observation="page b",
    )
    assert check.state_changed is True

    check2 = v.check(
        "click[Buy Now]",
        clickables=["buy now"],
        observation="page a",
        prev_observation="page a",
    )
    assert check2.state_changed is False


def test_extract_clickables():
    obs = (
        "WebShop [SEP] Instruction: [SEP] find a product [SEP] Back to Search "
        "[SEP] Next > [SEP] B000VOHH8I [SEP] A long product title that exceeds "
        "forty characters by quite a margin [SEP] 6.76 fl oz (pack of 1)"
    )
    clickables = extract_clickables_from_observation(obs)
    assert "back to search" in clickables
    assert "next >" in clickables
    assert "b000vohh8i" in clickables
    assert "6.76 fl oz (pack of 1)" in clickables
    assert "a long product title that exceeds forty characters by quite a margin" not in clickables


if __name__ == "__main__":
    import traceback

    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception:
                failures += 1
                print(f"FAIL {name}")
                traceback.print_exc()
    sys.exit(1 if failures else 0)
