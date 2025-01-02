import os
import gzip
import numpy as np
import xml.etree.ElementTree as ET
from typing import List
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import Model

###############################################################################
# 0) GPU Check
###############################################################################
def check_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("GPU device(s) found:")
        for gpu in gpus:
            print("  ", gpu)
    else:
        print("No GPU devices found. Running on CPU.")

###############################################################################
# 1) Parsing & Data-Building
###############################################################################
def seat_for_tag(tag: str) -> int:
    draw_map = {'T': 0, 'U': 1, 'V': 2, 'W': 3}
    discard_map = {'D': 0, 'E': 1, 'F': 2, 'G': 3}
    c = tag[0]
    if c in draw_map:
        return draw_map[c]
    elif c in discard_map:
        return discard_map[c]
    return -1

def read_mjlog(filename: str):
    try:
        with gzip.open(filename, "rb") as f:
            xml_data = f.read()
    except OSError:
        with open(filename, "rb") as f:
            xml_data = f.read()
    return ET.fromstring(xml_data.decode("utf-8"))

def remove_tiles_from_hand(hand: List[int], tile_ids: List[int]):
    for t in tile_ids:
        if t in hand:
            hand.remove(t)

def decode_meld_tiles(m_val: int) -> List[int]:
    meld_type = (m_val >> 3) & 0x07
    if meld_type == 0:  # chi
        base = (m_val >> 10) & 0x3F
        return [base, base+1, base+2]
    elif meld_type == 1:  # pon
        base = (m_val >> 9) & 0x7F
        base_tile = base & (~0x3)
        return [base_tile]*3
    elif meld_type in (3, 4, 5):  # kan
        if meld_type == 3:  # ankan
            base = (m_val >> 8) & 0xFF
            base_tile = base & (~0x3)
            return [base_tile]*4
        elif meld_type == 4:  # kakan
            base = (m_val >> 8) & 0xFF
            base_tile = base & (~0x3)
            return [base_tile]*4
        elif meld_type == 5:  # minkan
            base = (m_val >> 9) & 0x7F
            base_tile = base & (~0x3)
            return [base_tile]*4
    return []

def parse_mjlog_to_dataset(filename: str):
    root = read_mjlog(filename)
    if root is None:
        return []

    dataset = []
    round_index = -1

    seat_hands = {0: [], 1: [], 2: [], 3: []}
    discard_count = np.zeros((4, 136), dtype=int)

    in_round = False
    action_index = 0

    for elem in root:
        tag = elem.tag

        if tag == "INIT":
            round_index += 1
            in_round = True
            action_index = 0
            seat_hands = {0: [], 1: [], 2: [], 3: []}
            discard_count = np.zeros((4, 136), dtype=int)

            for s in range(4):
                attr = f"hai{s}"
                if attr in elem.attrib:
                    tiles_str = elem.attrib[attr]
                    tile_ids = [int(x) for x in tiles_str.split(",") if x.strip() != ""]
                    seat_hands[s] = tile_ids

        elif tag in ("AGARI", "RYUUKYOKU"):
            in_round = False
            continue

        if in_round:
            seat = seat_for_tag(tag)
            if seat != -1:
                tile_str = tag.lstrip('TUVWDEFG')
                if tile_str.isdigit():
                    tile_id = int(tile_str)
                    if tag[0] in "TUVW":
                        # Draw
                        seat_hands[seat].append(tile_id)
                    else:
                        # Discard
                        hand_before = seat_hands[seat][:]
                        opp_discards = np.sum(discard_count[[i for i in range(4) if i != seat]], axis=0)

                        row = {
                            'round_index': round_index,
                            'action_index': action_index,
                            'seat': seat,
                            'hand_before': hand_before,
                            'opponent_discards': opp_discards,
                            'discard_tile': tile_id
                        }
                        dataset.append(row)
                        action_index += 1

                        if tile_id in seat_hands[seat]:
                            seat_hands[seat].remove(tile_id)

                        discard_count[seat, tile_id] += 1
            elif tag == "N":
                who = int(elem.attrib["who"])
                m_val = int(elem.attrib["m"])
                meld_tiles = decode_meld_tiles(m_val)
                remove_tiles_from_hand(seat_hands[who], meld_tiles)

    return dataset

def parse_directory(dirpath: str):
    all_data = []
    for fname in os.listdir(dirpath):
        if fname.endswith('.mjlog') or fname.endswith('.mjlog.gz'):
            fullpath = os.path.join(dirpath, fname)
            print(f"Parsing: {fullpath}")
            file_data = parse_mjlog_to_dataset(fullpath)
            all_data.extend(file_data)

    print(f"Parsed {len(all_data)} discard events total from {dirpath}")
    return all_data

def encode_hand_to_136(hand_tiles: List[int]) -> np.ndarray:
    vec = np.zeros(136, dtype=np.float32)
    for t in hand_tiles:
        vec[t] += 1.0
    return vec

def build_dataset_rows(discard_rows):
    X_hand_list = []
    X_opp_list  = []
    y_list      = []

    for row in discard_rows:
        hand_vec = encode_hand_to_136(row['hand_before'])
        opp_vec  = row['opponent_discards'].astype(np.float32)
        discard  = row['discard_tile']

        X_hand_list.append(hand_vec)
        X_opp_list.append(opp_vec)
        y_list.append(discard)

    X_hand = np.stack(X_hand_list, axis=0)  # shape (N,136)
    X_opp  = np.stack(X_opp_list, axis=0)   # shape (N,136)
    y      = np.array(y_list, dtype=np.int32)

    return X_hand, X_opp, y

###############################################################################
# 2) Build Multi-Input Model
###############################################################################
def build_advanced_model():
    """
    Two-input model:
      - input_hand (136,)
      - input_opp (136,)
    with Tanh activations, Dropout, and final 'sigmoid' layer.
    """
    input_hand = layers.Input(shape=(136,), name='hand_input')
    x_hand = layers.Dense(1024, activation='relu')(input_hand)
    x_hand = layers.Dropout(0.2)(x_hand)
    x_hand = layers.Dense(1024, activation='relu')(x_hand)

    input_opp = layers.Input(shape=(136,), name='opp_discard_input')
    x_opp = layers.Dense(512, activation='relu')(input_opp)
    x_opp = layers.Dropout(0.2)(x_opp)
    x_opp = layers.Dense(512, activation='relu')(x_opp)

    merged = layers.Concatenate()([x_hand, x_opp])  # shape (512,)

    x = layers.Dense(1024, activation='relu')(merged)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.4)(x)

    # Output shape = 136
    output = layers.Dense(136, activation='softmax')(x)

    model = Model(inputs=[input_hand, input_opp], outputs=output)
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

###############################################################################
# 3) Cross-Validation for Multi-Input
###############################################################################
def cross_validate_model(X_hand, X_opp, y, n_splits=3, epochs=10, batch_size=64):
    """
    Perform KFold cross-validation on the multi-input dataset:
     - X_hand: (N,136)
     - X_opp:  (N,136)
     - y:      (N,)

    We'll build a fresh model each fold, then train/eval on that fold.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    accuracies = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X_hand), start=1):
        print(f"\n=== Fold {fold}/{n_splits} ===")
        # Split data
        X_hand_train, X_hand_test = X_hand[train_idx], X_hand[test_idx]
        X_opp_train,  X_opp_test  = X_opp[train_idx],  X_opp[test_idx]
        y_train,      y_test      = y[train_idx],      y[test_idx]

        # Build fresh model
        model = build_advanced_model()

        # Train
        model.fit(
            [X_hand_train, X_opp_train], y_train,
            validation_data=([X_hand_test, X_opp_test], y_test),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )

        # Evaluate
        loss, acc = model.evaluate([X_hand_test, X_opp_test], y_test, verbose=0)
        accuracies.append(acc)
        print(f"Fold {fold} accuracy: {acc:.4f}")

    avg_acc = np.mean(accuracies)
    print(f"\nCross-validation average accuracy: {avg_acc:.4f}")

    return avg_acc

###############################################################################
# 4) Main
###############################################################################
def main():
    check_gpu()

    dataset_file = "mjai_dataset.npz"  # Name for cached dataset
    if os.path.exists(dataset_file):
        # Load dataset from file
        print(f"Loading dataset from {dataset_file} ...")
        data = np.load(dataset_file)
        X_hand = data['X_hand']
        X_opp  = data['X_opp']
        y      = data['y']
        print("Loaded dataset shapes:",
              X_hand.shape, X_opp.shape, y.shape)
    else:
        # Parse logs and build dataset
        dirpath = "./mjlog_dataset"  # Adjust if needed
        all_data = parse_directory(dirpath)
        X_hand, X_opp, y = build_dataset_rows(all_data)
        np.savez(dataset_file, X_hand=X_hand, X_opp=X_opp, y=y)
        print(f"Dataset saved to {dataset_file}")

    # (Optional) CROSS-VALIDATION
    # Uncomment to run cross-validation
    # cross_validate_model(X_hand, X_opp, y, n_splits=3, epochs=5, batch_size=64)

    # TRAIN/TEST SPLIT
    X_hand_train, X_hand_test, X_opp_train, X_opp_test, y_train, y_test = train_test_split(
        X_hand, X_opp, y, test_size=0.2, random_state=42
    )
    print("Train/Test split done.")
    print("Train shapes:", X_hand_train.shape, X_opp_train.shape, y_train.shape)
    print("Test  shapes:", X_hand_test.shape,  X_opp_test.shape,  y_test.shape)

    # BUILD MODEL
    model = build_advanced_model()
    model.summary()

    # TRAIN
    model.fit(
        [X_hand_train, X_opp_train],
        y_train,
        validation_data=([X_hand_test, X_opp_test], y_test),
        epochs=20,
        batch_size=128
    )

    # EVALUATE
    loss, acc = model.evaluate([X_hand_test, X_opp_test], y_test)
    print(f"Test Accuracy = {acc:.4f}")

    # SAVE
    model.save("my_mahjong_model.h5")
    print("Model saved to my_mahjong_model.h5")

if __name__ == "__main__":
    main()
