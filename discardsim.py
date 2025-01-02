import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suprimir mensagens INFO e WARNING do TensorFlow
import numpy as np
import tensorflow as tf
from mahjong.shanten import Shanten
from mahjong.tile import TilesConverter

###############################################################################
# 1) Tile ID ↔ Mahjong Unicode Icons
###############################################################################

def tile_id_to_unicode(tile_id: int) -> str:
    """
    Converte tile_id (0..135) para o caractere Unicode correspondente do Mahjong.
    """
    if not (0 <= tile_id <= 135):
        return f"[Inválido {tile_id}]"

    # Manzu
    if 0 <= tile_id <= 35:
        rank = tile_id // 4 + 1
        return chr(0x1F007 + (rank - 1))  # 🀇..🀏
    # Pinzu
    elif 36 <= tile_id <= 71:
        rank = (tile_id - 36)//4 + 1
        return chr(0x1F019 + (rank - 1))  # 🀙..🀡
    # Souzu
    elif 72 <= tile_id <= 107:
        rank = (tile_id - 72)//4 + 1
        return chr(0x1F010 + (rank - 1))  # 🀐..🀘
    # Honors
    else:
        group = (tile_id - 108)//4
        honors = ['🀀','🀁','🀂','🀃','🀆','🀅','🀄']
        if 0 <= group < len(honors):
            return honors[group]
        return f"[Honors Inválido {tile_id}]"

###############################################################################
# 2) Tile ID ↔ User Codes (como "1m", "5p", "E", etc.)
###############################################################################

def user_code_to_tile_ids(user_code: str):
    """
    Converte strings como '1m' => tile IDs [0..3]
                  '9s' => tile IDs [104..107]
                  'E' => tile IDs [108..111]
                  ...
    Retorna uma lista de todos os tile IDs que correspondem ao código.
    Se inválido, retorna [].
    """
    user_code = user_code.strip().upper()  # Ex.: "1m" -> "1M"
    # Para suits: ex.: '5m' => rank=5, suit='m'
    if len(user_code) == 2 and user_code[0].isdigit():
        rank_char = user_code[0]        # '1'..'9'
        suit_char = user_code[1].lower()# 'm','p','s'
        if rank_char.isdigit() and suit_char in ['m','p','s']:
            rank = int(rank_char)
            if 1 <= rank <= 9:
                # Converter para tile IDs
                # Manzu => 0..35, Pinzu =>36..71, Souzu=>72..107
                suit_map = {'m':0,'p':1,'s':2}
                suit_idx = suit_map[suit_char]
                base = suit_idx*36 + (rank-1)*4
                return list(range(base, base+4))
        return []
    elif len(user_code) == 1:
        # Pode ser um honor como E,S,W,N,H,G,R ?
        c = user_code[0]
        honor_map = {
            'E':108, 'S':112, 'W':116, 'N':120,
            'H':124, # Haku (white)
            'G':128, # Hatsu (green)
            'R':132, # Chun (red)
        }
        if c in honor_map:
            base = honor_map[c]
            return list(range(base, base+4))
        return []
    return []

def tile_id_to_user_code(tile_id: int) -> str:
    """
    Converte um único tile_id para o código correspondente, ex.: "1m","9p","E" etc.
    """
    if 0 <= tile_id <= 107:
        suit_idx = tile_id // 36        # 0=man,1=pin,2=sou
        rank_0_to_8 = (tile_id % 36)//4 # 0..8 => rank=1..9
        rank = rank_0_to_8 + 1
        suits = ['m','p','s']
        return f"{rank}{suits[suit_idx]}"
    else:
        # honors
        group = (tile_id - 108)//4  # 0..6
        codes = ['E','S','W','N','H','G','R']
        if 0<=group<len(codes):
            return codes[group]
        return "?"

###############################################################################
# 3) Hand Encoding & Shanten Check (Usando python-mahjong)
###############################################################################

def encode_hand_136(hand_ids):
    """
    Retorna um vetor multi-hot de 136 dimensões para a mão.
    """
    arr = np.zeros(136, dtype=np.float32)
    for t in hand_ids:
        if 0 <= t < 136:
            arr[t] += 1
    return arr

def tile_id_to_34(tile_id):
    """
    Converte um tile ID (0..135) para uma representação de 34-tile para a biblioteca de Mahjong.
    """
    # Manzu: 0-35 -> 0-8
    # Pinzu: 36-71 -> 9-17
    # Souzu: 72-107 -> 18-26
    # Honors: 108-135 -> 27-33
    if 0 <= tile_id <= 35:
        return tile_id // 4  # 0-8
    elif 36 <= tile_id <= 71:
        return 9 + ((tile_id - 36) // 4)  # 9-17
    elif 72 <= tile_id <= 107:
        return 18 + ((tile_id - 72) // 4)  # 18-26
    elif 108 <= tile_id <= 135:
        return 27 + ((tile_id - 108) // 4)  # 27-33
    else:
        raise ValueError(f"Invalid tile_id: {tile_id}")

def calc_shanten(hand_tile_ids):
    """
    Calcula o número de shanten usando python-mahjong.
    """
    shanten_calculator = Shanten()
    # Converter 136-tile IDs para representação de 34-tile
    counts_34 = [0] * 34
    for t in hand_tile_ids:
        counts_34[tile_id_to_34(t)] += 1
    shanten_number = shanten_calculator.calculate_shanten(counts_34)
    return shanten_number

def is_tenpai(hand_tile_ids):
    """
    Determina se a mão está em tenpai (shanten <= 1).
    """
    shanten = calc_shanten(hand_tile_ids)
    return False

###############################################################################
# 4) Funções Auxiliares
###############################################################################

def show_hand_icons(hand_ids):
    """
    Ordena por tile_id e imprime os ícones.
    """
    sorted_tiles = sorted(hand_ids)
    icons = [tile_id_to_unicode(t) for t in sorted_tiles]
    hand_str = " ".join(icons)
    print(f"Hand => {hand_str}")

def pick_first_in_hand(possible_ids, hand):
    """
    Entre os tile IDs candidatos para aquele código de usuário, pega o primeiro
    que está realmente na 'hand'. Se nenhum => retorna -1.
    """
    for tid in possible_ids:
        if tid in hand:
            return tid
    return -1

###############################################################################
# 5) Treinador Interativo Atualizado
###############################################################################

def main():
    # A) Carregar o modelo de descarte multi-input
    model_path = "my_mahjong_model.h5"
    model = tf.keras.models.load_model(model_path)
    print(f"Loaded model from {model_path}\n")
    model.summary()  # Opcional: para confirmar a arquitetura do modelo

    # B) Embaralhar uma parede de 136 tiles e distribuir 13 tiles
    wall = list(range(136))
    np.random.shuffle(wall)
    hand = wall[:13]
    wall_index = 13

    while True:
        # 1) Puxar um tile (totalizando 14)
        if wall_index >= 136:
            print("Wall exhausted. Stopping.")
            break
        drawn_tile = wall[wall_index]
        wall_index += 1
        hand.append(drawn_tile)

        # Exibir o tile puxado
        drawn_code = tile_id_to_user_code(drawn_tile)
        drawn_icon = tile_id_to_unicode(drawn_tile)
        print(f"\nYou drew => {drawn_code} ({drawn_icon})")

        # 2) Verificar se está em tenpai
        if is_tenpai(hand):
            print("\n*** You are in TENPAI! ***")
            show_hand_icons(hand)
            break

        # 3) Exibir a mão atual (ícones)
        print("\n--- Current 14-tile Hand (icons) ---")
        show_hand_icons(hand)

        # 4) Gerar e exibir as top 5 recomendações do modelo
        X_hand = encode_hand_136(hand)  # shape (136,)
        X_opp = np.zeros(136, dtype=np.float32)  # Dummy opponent input

        X_input = [X_hand[np.newaxis, :], X_opp[np.newaxis, :]]  # shape (1,136), (1,136)
        preds = model.predict(X_input)[0]  # shape (136,)

        # Ordenar por probabilidade decrescente
        sorted_ids = np.argsort(preds)[::-1]
        top_5_ids = sorted_ids[:5]
        print("\nModel's top 5 recommended discards:")
        recommended_tiles = []
        for i_top in range(5):
            tid = sorted_ids[i_top]
            prob = preds[tid]
            code = tile_id_to_user_code(tid)
            recommended_tiles.append(tid)
            print(f"  {code} => {prob:.3f}")

        # 5) Perguntar ao usuário para descartar por código de tile
        user_input = input("Discard tile code (e.g. '5m', 'E') or 'q' to quit:\n> ").strip()
        if user_input.lower() in ('q','quit','exit'):
            print("Exiting trainer. Bye!")
            break

        # 6) Encontrar o tile na mão
        candidate_ids = user_code_to_tile_ids(user_input)
        discard_tile = pick_first_in_hand(candidate_ids, hand)
        if discard_tile < 0:
            print(f"You don't have '{user_input}' in your hand. Try again.")
            continue

        # 7) Remover o tile escolhido da mão
        hand.remove(discard_tile)
        disc_code = tile_id_to_user_code(discard_tile)
        disc_icon = tile_id_to_unicode(discard_tile)
        print(f"YOU discarded => {disc_code} ({disc_icon})")

        # 8) Verificar se o tile descartado estava entre as recomendações
        if discard_tile in top_5_ids:
            print("Good choice! You selected one of the top recommended discards.")
        else:
            print("You selected a discard outside the top recommendations.")

        # Próxima iteração => puxar novamente

    print("\nDone. Thanks for using the MahjML v1.0 trainer with icons & codes!")

###############################################################################
# 6) Execução do Programa
###############################################################################

if __name__=="__main__":
    main()
