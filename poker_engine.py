"""本地德州扑克牌力与胜率计算；不依赖网络或外部牌局服务。"""
import itertools
import random
import re

SUIT_SYMBOLS = {"S": "♠", "H": "♥", "D": "♦", "C": "♣"}
SUIT_ALIASES = {
    "S": "S", "♠": "S", "SPADES": "S",
    "H": "H", "♥": "H", "HEARTS": "H",
    "D": "D", "♦": "D", "DIAMONDS": "D",
    "C": "C", "♣": "C", "CLUBS": "C",
}
RANK_VALUES = {**{str(n): n for n in range(2, 10)}, "T": 10, "10": 10, "J": 11, "Q": 12, "K": 13, "A": 14}
RANK_NAMES = {**{n: str(n) for n in range(2, 10)}, 10: "T", 11: "J", 12: "Q", 13: "K", 14: "A"}
CATEGORY_NAMES = {
    8: "同花顺", 7: "四条", 6: "葫芦", 5: "同花",
    4: "顺子", 3: "三条", 2: "两对", 1: "一对", 0: "高牌",
}
CARD_RE = re.compile(r"^(10|[2-9TJQKA])\s*(SPADES|HEARTS|DIAMONDS|CLUBS|[SHDC♠♥♦♣?])$", re.IGNORECASE)

def parse_card(card):
    text = str(card).strip().upper().replace(" ", "")
    match = CARD_RE.fullmatch(text)
    if not match:
        raise ValueError(f"无法识别的牌：{card}")
    rank_text, suit_text = match.groups()
    suit = None if suit_text == "?" else SUIT_ALIASES[suit_text.upper()]
    return RANK_VALUES[rank_text.upper()], suit

def parse_cards(cards, allow_unknown=False):
    parsed = [parse_card(card) for card in cards]
    if not allow_unknown and any(suit is None for _, suit in parsed):
        raise ValueError("花色未识别")
    known = [(rank, suit) for rank, suit in parsed if suit is not None]
    if len(set(known)) != len(known):
        raise ValueError("检测到重复牌")
    return parsed

def _straight_high(ranks):
    unique = set(ranks)
    if 14 in unique:
        unique.add(1)
    for high in range(14, 4, -1):
        if all(rank in unique for rank in range(high - 4, high + 1)):
            return high
    return None

def _evaluate_five(cards):
    ranks = [rank for rank, _ in cards]
    suits = [suit for _, suit in cards]
    counts = {}
    for rank in ranks:
        counts[rank] = counts.get(rank, 0) + 1
    groups = sorted(((count, rank) for rank, count in counts.items()), reverse=True)
    flush = len(set(suits)) == 1
    straight = _straight_high(ranks)

    if flush and straight:
        return (8, straight)
    if groups[0][0] == 4:
        quad = groups[0][1]
        kicker = max(rank for rank in ranks if rank != quad)
        return (7, quad, kicker)
    if groups[0][0] == 3 and groups[1][0] == 2:
        return (6, groups[0][1], groups[1][1])
    if flush:
        return (5, *sorted(ranks, reverse=True))
    if straight:
        return (4, straight)
    if groups[0][0] == 3:
        trip = groups[0][1]
        kickers = sorted((rank for rank in ranks if rank != trip), reverse=True)
        return (3, trip, *kickers)
    pairs = sorted((rank for rank, count in counts.items() if count == 2), reverse=True)
    if len(pairs) == 2:
        kicker = max(rank for rank, count in counts.items() if count == 1)
        return (2, pairs[0], pairs[1], kicker)
    if len(pairs) == 1:
        pair = pairs[0]
        kickers = sorted((rank for rank in ranks if rank != pair), reverse=True)
        return (1, pair, *kickers)
    return (0, *sorted(ranks, reverse=True))

def evaluate_hand(cards):
    parsed = parse_cards(cards)
    if not 5 <= len(parsed) <= 7:
        raise ValueError("牌型计算需要 5 到 7 张有效牌")
    best = max(_evaluate_five(combo) for combo in itertools.combinations(parsed, 5))
    return best, best[0], CATEGORY_NAMES[best[0]]

def _card_text(card):
    rank, suit = card
    return f"{RANK_NAMES[rank]}{SUIT_SYMBOLS[suit]}"

def estimate_equity(hole_cards, board_cards, iterations=400, seed=None):
    """估算对随机对手的胜率；未知花色按剩余花色随机积分，平局按半胜计。"""
    if len(hole_cards) != 2 or len(board_cards) > 5:
        raise ValueError("需要两张手牌，公共牌不能超过五张")
    known = parse_cards(list(hole_cards) + list(board_cards), allow_unknown=True)
    fixed_suits = {(rank, suit) for rank, suit in known if suit is not None}
    unknown_by_rank = {}
    for index, (rank, suit) in enumerate(known):
        if suit is None:
            unknown_by_rank.setdefault(rank, []).append(index)
    choices = {}
    for rank, indexes in unknown_by_rank.items():
        available = [suit for suit in "SHDC" if (rank, suit) not in fixed_suits]
        if len(available) < len(indexes):
            raise ValueError("同一点数的牌无法分配到不同花色")
        choices[rank] = available

    missing_board = 5 - len(board_cards)
    needed = missing_board + 2
    rng = random.Random(seed)
    wins = 0.0
    sample_count = max(1, int(iterations))
    for _ in range(sample_count):
        assigned = list(known)
        for rank, indexes in unknown_by_rank.items():
            suits = rng.sample(choices[rank], len(indexes))
            for index, suit in zip(indexes, suits):
                assigned[index] = (rank, suit)
        unavailable = set(assigned)
        deck = [(rank, suit) for rank in range(2, 15) for suit in "SHDC" if (rank, suit) not in unavailable]
        drawn = rng.sample(deck, needed)
        hero = [_card_text(card) for card in assigned[:2]]
        known_board = [_card_text(card) for card in assigned[2:]]
        future_board = known_board + [_card_text(card) for card in drawn[:missing_board]]
        opponent = [_card_text(card) for card in drawn[missing_board:]]
        hero_score = evaluate_hand(hero + future_board)[0]
        opponent_score = evaluate_hand(opponent + future_board)[0]
        wins += 1.0 if hero_score > opponent_score else 0.5 if hero_score == opponent_score else 0.0
    return {"equity": wins / sample_count, "samples": sample_count, "unknown_suits": sum(map(len, unknown_by_rank.values()))}

def self_check():
    royal = evaluate_hand(["AS", "KS", "QS", "JS", "10S"])
    wheel = evaluate_hand(["AS", "2S", "3S", "4S", "5S"])
    if royal[1] != 8 or royal[0][1] != 14 or wheel[0][1] != 5:
        raise RuntimeError("牌型计算自检失败")
    equity = estimate_equity(["AS", "A?"], ["AD", "AC", "2S"], iterations=80, seed=17)
    if not 0.8 <= equity["equity"] <= 1.0 or equity["unknown_suits"] != 1:
        raise RuntimeError("胜率计算自检失败")
    return True
