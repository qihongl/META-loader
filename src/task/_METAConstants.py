from utils import reverse_dict

hier_struct = {
    "exercising": {
        "performing_cardio_exercises": [
            "jump_rope",
            "perform_jumping_jacks",
            "do_stair_steps"
        ],
        "performing_resistance_exercises": [
            "bicep_curls",
            "shoulder_press",
            "push_ups",
            "sit_ups"
        ],
        "taking_a_snack_break": [
            "drink_sport_drink",
            "eat_a_granola_bar",
            "drink_water"
        ]
    },
    "making_breakfast": {
        "preparing_main_items": [
            "prepare_cereal",
            "prepare_a_bagel",
            "prepare_hot_oatmeal"
        ],
        "preparing_side_items": [
            "prepare_fresh_fruit",
            "prepare_toast",
            "prepare_yogurt_with_granola"
        ],
        "preparing_beverages": [
            "prepare_orange_juice",
            "prepare_tea",
            "prepare_instant_coffee",
            "prepare_milk"
        ]
    },
    "cleaning_a_room": {
        "removing_dust": [
            "vacuum_floor",
            "use_hand_duster",
            "use_vacuum_attachment"
        ],
        "preparing_a_bed": [
            "put_on_bed_sheets",
            "put_cases_on_pillows",
            "fold_blanket_or_comforter"
        ],
        "folding_laundry": [
            "fold_shirts_or_pants",
            "fold_socks",
            "fold_towels"
        ]
    },
    "bathroom_grooming": {
        "grooming_mouth": [
            "brush_teeth",
            "floss_teeth",
            "use_mouthwash",
            "apply_chapstick"
        ],
        "grooming_hair": [
            "blowdry_hair",
            "brush_hair",
            "comb_hair",
            "use_hair_gel"
        ],
        "grooming_face": [
            "wash_face",
            "shave_face",
            "apply_lotion"
        ]
    },
    "multichapter_actions": [
        "take_a_pill",
        "torso_rotations",
        "put_objects_in_drawers",
        "take_objects_out_of_drawers",
        "clean_a_surface",
        "look_at_text_message"
    ]
}

hlc2hlcid = {
    'multichapter_actions':0,
    'exercising':1,
    'making_breakfast':2,
    'cleaning_a_room':3,
    'bathroom_grooming':4,
}

mlc2mlcid = {
    'performing_cardio_exercises':0,
    'performing_resistance_exercises':1,
    'taking_a_snack_break':2,
    'preparing_main_items':3,
    'preparing_side_items':4,
    'preparing_beverages':5,
    'removing_dust':6,
    'preparing_a_bed':7,
    'folding_laundry':8,
    'grooming_mouth':9,
    'grooming_hair':10,
    'grooming_face:':11,
}


llc_names = ['apply_chapstick', 'apply_lotion', 'bicep_curls', 'blowdry_hair',
'brush_hair', 'brush_teeth', 'clean_a_surface', 'comb_hair', 'do_stair_steps',
'drink_sport_drink', 'drink_water', 'eat_a_granola_bar', 'floss_teeth',
'fold_blanket_or_comforter', 'fold_shirts_or_pants', 'fold_socks', 'fold_towels',
'jump_rope', 'look_at_text_message', 'perform_jumping_jacks', 'prepare_a_bagel',
'prepare_cereal', 'prepare_fresh_fruit', 'prepare_hot_oatmeal',
'prepare_instant_coffee', 'prepare_milk', 'prepare_orange_juice', 'prepare_tea',
'prepare_toast', 'prepare_yogurt_with_granola', 'push_ups', 'put_cases_on_pillows',
'put_objects_in_drawers', 'put_on_bed_sheets', 'shave_face', 'shoulder_press',
'sit_ups', 'take_a_pill', 'take_objects_out_of_drawers', 'torso_rotations',
'use_hair_gel', 'use_hand_duster', 'use_mouthwash', 'use_vacuum_attachment',
'vacuum_floor', 'wash_face']

llc2llcid = {llc_i:i for i, llc_i in enumerate(llc_names)}

llc2hlcid = {
    'apply_chapstick': 4,
    'apply_lotion': 4,
    'bicep_curls': 1,
    'blowdry_hair': 4,
    'brush_hair': 4,
    'brush_teeth': 4,
    'clean_a_surface': 0,
    'comb_hair': 4,
    'do_stair_steps': 1,
    'drink_sport_drink': 1,
    'drink_water': 1,
    'eat_a_granola_bar': 1,
    'floss_teeth': 4,
    'fold_blanket_or_comforter': 3,
    'fold_shirts_or_pants': 3,
    'fold_socks': 3,
    'fold_towels': 3,
    'jump_rope': 1,
    'look_at_text_message': 0,
    'perform_jumping_jacks': 1,
    'prepare_a_bagel': 2,
    'prepare_cereal': 2,
    'prepare_fresh_fruit': 2,
    'prepare_hot_oatmeal': 2,
    'prepare_instant_coffee': 2,
    'prepare_milk': 2,
    'prepare_orange_juice': 2,
    'prepare_tea': 2,
    'prepare_toast': 2,
    'prepare_yogurt_with_granola': 2,
    'push_ups': 1,
    'put_cases_on_pillows': 3,
    'put_objects_in_drawers': 0,
    'put_on_bed_sheets': 3,
    'shave_face': 4,
    'shoulder_press': 1,
    'sit_ups': 1,
    'take_a_pill': 0,
    'take_objects_out_of_drawers': 0,
    'torso_rotations': 0,
    'use_hair_gel': 4,
    'use_hand_duster': 3,
    'use_mouthwash': 4,
    'use_vacuum_attachment': 3,
    'vacuum_floor': 3,
    'wash_face': 4
}


llc2mlcid = {
    'apply_chapstick': 9,
    'apply_lotion': 11,
    'bicep_curls': 1,
    'blowdry_hair': 10,
    'brush_hair': 10,
    'brush_teeth': 9,
    'clean_a_surface': None,
    'comb_hair': 10,
    'do_stair_steps': 0,
    'drink_sport_drink': 2,
    'drink_water': 2,
    'eat_a_granola_bar': 2,
    'floss_teeth': 9,
    'fold_blanket_or_comforter': 7,
    'fold_shirts_or_pants': 8,
    'fold_socks': 8,
    'fold_towels': 8,
    'jump_rope': 0,
    'look_at_text_message': None,
    'perform_jumping_jacks': 0,
    'prepare_a_bagel': 3,
    'prepare_cereal': 3,
    'prepare_fresh_fruit': 4,
    'prepare_hot_oatmeal': 3,
    'prepare_instant_coffee': 5,
    'prepare_milk': 5,
    'prepare_orange_juice': 5,
    'prepare_tea': 5,
    'prepare_toast': 4,
    'prepare_yogurt_with_granola': 4,
    'push_ups': 1,
    'put_cases_on_pillows': 7,
    'put_objects_in_drawers': None,
    'put_on_bed_sheets': 7,
    'shave_face': 11,
    'shoulder_press': 1,
    'sit_ups': 1,
    'take_a_pill': None,
    'take_objects_out_of_drawers': None,
    'torso_rotations': None,
    'use_hair_gel': 10,
    'use_hand_duster': 6,
    'use_mouthwash': 9,
    'use_vacuum_attachment': 6,
    'vacuum_floor': 6,
    'wash_face': 11
}


llcid2llc = reverse_dict(llc2llcid)
llcid2hlcid = {k:llc2hlcid[v] for k,v in llcid2llc.items()}
llcid2mlcid = {k:llc2mlcid[v] for k,v in llcid2llc.items()}
