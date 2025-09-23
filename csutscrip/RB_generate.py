import numpy as np

"""
Define matrices representing (all) Clifford gates for single
qubit in the basis of Z, X, Y, -Z, -X, -Y, indicating
where on the 6 cardinal points of the Bloch sphere the
+Z, +X, +Y axes go after each gate. Each Clifford gate
can be uniquely identified just by checking where +X and +Y
go.
"""
clifford_1q = dict()
# clifford_1q["Z"] = np.matrix(
#     [
#         [1, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 1, 0],
#         [0, 0, 0, 0, 0, 1],
#         [0, 0, 0, 1, 0, 0],
#         [0, 1, 0, 0, 0, 0],
#         [0, 0, 1, 0, 0, 0],
#     ]
# )
clifford_1q["X"] = np.matrix(
    [
        [0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 0],
    ]
)
clifford_1q["Y"] = np.matrix(
    [
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1],
    ]
)
# clifford_1q["Z/2"] = np.matrix(
#     [
#         [1, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 1],
#         [0, 1, 0, 0, 0, 0],
#         [0, 0, 0, 1, 0, 0],
#         [0, 0, 1, 0, 0, 0],
#         [0, 0, 0, 0, 1, 0],
#     ]
# )
clifford_1q["X/2"] = np.matrix(
    [
        [0, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 0],
    ]
)
clifford_1q["Y/2"] = np.matrix(
    [
        [0, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1],
    ]
)
# clifford_1q["-Z/2"] = np.matrix(
#     [
#         [1, 0, 0, 0, 0, 0],
#         [0, 0, 1, 0, 0, 0],
#         [0, 0, 0, 0, 1, 0],
#         [0, 0, 0, 1, 0, 0],
#         [0, 0, 0, 0, 0, 1],
#         [0, 1, 0, 0, 0, 0],
#     ]
# )
clifford_1q["-X/2"] = np.matrix(
    [
        [0, 0, 0, 0, 0, 1],
        [0, 1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0],
    ]
)
clifford_1q["-Y/2"] = np.matrix(
    [
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1],
    ]
)
identity = np.diag([1] * 6)
clifford_1q["I"] = identity


# Read pulse as a matrix product acting on state (meaning apply pulses in reverse order of the tuple)
# two_step_pulses = [
#     ("X", "Z/2"),
#     ("X/2", "Z/2"),
#     ("-X/2", "Z/2"),
#     ("Y", "Z/2"),
#     ("Y/2", "Z/2"),
#     ("-Y/2", "Z/2"),
#     ("X", "Z"),
#     ("X/2", "Z"),
#     ("-X/2", "Z"),
#     ("Y", "Z"),
#     ("Y/2", "Z"),
#     ("-Y/2", "Z"),
#     ("X", "-Z/2"),
#     ("X/2", "-Z/2"),
#     ("-X/2", "-Z/2"),
#     ("Y", "-Z/2"),
#     ("Y/2", "-Z/2"),
#     ("-Y/2", "-Z/2"),
# ]

step_pulses = [
    ("Y/2", "X"),
    ("Y/2", "X/2"),
    ("X/2", "-Y/2", "-X/2"),
    ("-X/2", "-Y/2"),
    ("Y/2", "-X/2"),
    ("-X/2", "Y/2", "-X/2"),
    ("X/2", "Y/2"),
    ("-Y/2", "X"),
    ("-X/2", "Y"),
    ("-Y/2", "-X/2"),
    ("X/2", "Y/2", "X/2"),
    ("-X/2", "Y/2"),
    ("X", "Y"),
    ("X/2", "Y"),
    ("-Y/2", "X/2"),
    ("X/2", "Y/2", "-X/2"),
    ("X/2", "-Y/2"),
]

for pulse in step_pulses:
    new_mat = clifford_1q[pulse[0]]
    for p in pulse[1:]:
        new_mat = new_mat @ clifford_1q[p]
    repeat = False
    # Make sure there are no repeats
    for existing_pulse_name, existing_pulse in clifford_1q.items():
        if np.array_equal(new_mat, existing_pulse):
            print("found repeat", pulse, existing_pulse_name)
            repeat = True
    if not repeat:
        clifford_1q[pulse[0] + "," + ",".join(pulse[1:])] = new_mat
clifford_1q_names = list(clifford_1q.keys())
assert (
    len(clifford_1q_names) == 24
), f"you have {len(clifford_1q_names)} elements in your Clifford group instead of 24!"
# print(len(clifford_1q_names), "elements in clifford_1q")
# print(clifford_1q_names)

# Get the average number of X/2 gates per Clifford gate
count = 0
for n in range(len(clifford_1q_names)):  # n is index in clifford_1q_names
    gates = clifford_1q_names[n].split(",")
    for gate in gates:
        # print(gate)
        if gate == "I" or "Z" in gate:
            continue
        if "/2" in gate:
            count += 1
            # print("added 1 to count")
        else:
            count += 2
            # print("added 2 to count")
# print("Average number of X/2 gates per Clifford gate:", count / len(clifford_1q_names))

for name, matrix in clifford_1q.items():
    z_new = np.argmax(matrix[:, 0])  # +Z goes to row where col 0 is 1
    x_new = np.argmax(matrix[:, 1])  # +X goes to row where col 1 is 1
    # print(name, z_new, x_new)
    clifford_1q[name] = (matrix, (z_new, x_new))


def gate_sequence(rb_depth, pulse_n_seq=None, debug=False):
    """
    Generate RB forward gate sequence of length rb_depth as a list of pulse names;
    also return the Clifford gate that is equivalent to the total pulse sequence.
    The effective inverse is pi phase + the total Clifford.
    Optionally, provide pulse_n_seq which is a list of the indices of the Clifford
    gates to apply in the sequence.
    """
    if pulse_n_seq is None:
        pulse_n_seq = (len(clifford_1q_names) * np.random.rand(rb_depth)).astype(int)
    pulse_name_seq = [clifford_1q_names[n] for n in pulse_n_seq]
    if debug:
        print("pulse seq", pulse_name_seq)
    psi_nz = np.matrix([[1, 0, 0, 0, 0, 0]]).transpose()
    psi_nx = np.matrix([[0, 1, 0, 0, 0, 0]]).transpose()
    for n in pulse_n_seq:  # n is index in clifford_1q_names
        gates = clifford_1q_names[n].split(",")
        for gate in reversed(gates):  # Apply matrices from right to left of gates
            psi_nz = clifford_1q[gate][0] @ psi_nz
            psi_nx = clifford_1q[gate][0] @ psi_nx
    psi_nz = psi_nz.flatten()
    psi_nx = psi_nx.flatten()
    if debug:
        print("+Z axis after seq:", psi_nz, "+X axis after seq:", psi_nx)

    total_clifford = None
    if np.argmax(psi_nz) == 0:
        total_clifford = "I"
    else:
        for clifford in clifford_1q_names:  # Get the clifford equivalent to the total seq
            if clifford_1q[clifford][1] == (np.argmax(psi_nz), np.argmax(psi_nx)):
                # z_new, x_new = clifford_1q[clifford][1]
                # if z_new == np.argmax(psi_nz):
                total_clifford = clifford
                break
    assert total_clifford is not None, f"Failed to invert gate sequence! {pulse_name_seq} which brings +Z to {psi_nz}"

    if debug:
        total_clifford_mat = clifford_1q[total_clifford][0]
        print("Total gate matrix:\n", total_clifford_mat)

    return pulse_name_seq, total_clifford


def interleaved_gate_sequence(rb_depth, gate_char: str, debug=False):
    """
    Generate RB gate sequence with rb_depth random gates interleaved with gate_char
    Returns the total gate list (including the interleaved gates) and the total
    Clifford gate equivalent to the total pulse sequence.
    """
    pulse_n_seq_rand = (len(clifford_1q_names) * np.random.rand(rb_depth)).astype(int)
    pulse_n_seq = []
    assert gate_char in clifford_1q_names
    n_gate_char = clifford_1q_names.index(gate_char)
    if debug:
        print("n gate char:", n_gate_char, clifford_1q_names[n_gate_char])
    for n_rand in pulse_n_seq_rand:
        pulse_n_seq.append(n_rand)
        pulse_n_seq.append(n_gate_char)
    return gate_sequence(len(pulse_n_seq), pulse_n_seq=pulse_n_seq, debug=debug)

def expand_full_sequence(pulse_name_seq, total_clifford):
    """
    Expand a full pulse_name_seq into the actual flat play sequence, 
    handling normal and inverse gates correctly.

    Args:
        pulse_name_seq : list of str
            e.g., ['X/2', 'X/2', 'X/2,-Y/2,-X/2', 'X/2']
        total_clifford : str
            e.g., 'Y/2,-X/2'

    Returns:
        list of str
            Final flat sequence to be played on hardware, with each gate separately listed
    """
    full_sequence = []

    # Expand normal pulse names (right-to-left)
    for name in pulse_name_seq:
        gates = name.split(",")
        for g in reversed(gates):   # right-to-left
            full_sequence.append(g)

    # Expand total_clifford separately (left-to-right, inverse)
    for gate in total_clifford.split(","):
        neg = "-" in gate
        neg = not neg  # inverse要翻轉正負號

        if neg:
            if "-" not in gate:
                gate = "-" + gate
        else:
            if "-" in gate:
                gate = gate.replace("-", "")

        full_sequence.append(gate)

    # 最後做一次展平：如果某個gate裡還有逗號，就切開
    final_sequence = []
    for item in full_sequence:
        if "," in item:
            final_sequence.extend(item.split(","))
        else:
            final_sequence.append(item)

    return final_sequence


if __name__ == "__main__":
    print("Clifford gates:", clifford_1q_names)
    print("Total number Clifford gates:", len(clifford_1q_names))
    pulse_name_seq, total_clifford = gate_sequence(2, debug=True)
    print("Pulse sequence:", pulse_name_seq)
    print("Total clifford of seq:", total_clifford)
    print("Operate gate:", expand_full_sequence(pulse_name_seq, total_clifford))

    gate_char = "X/2"
    print()
    print("Interleaved RB with gate", gate_char)
    pulse_name_seq, total_clifford = interleaved_gate_sequence(2, gate_char=gate_char, debug=True)
    print("Pulse sequence:", pulse_name_seq)
    print("Total clifford of seq:", total_clifford)
    pulse_name_seq.append(total_clifford)
    print("Operate gate:", expand_full_sequence(pulse_name_seq, total_clifford))