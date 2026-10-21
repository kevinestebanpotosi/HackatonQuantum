import json
import pennylane as qp
import pennylane.numpy as np


def prepare_and_build_checks(n):
    """Prepare an n-qubit GHZ state and return its stabilizer checks."""
    # Prepare |GHZ_n> = (|00...0> + |11...1>) / sqrt(2).
    qp.Hadamard(wires=0)
    for wire in range(1, n):
        qp.CNOT(wires=[0, wire])

    # Global X parity: X_0 X_1 ... X_(n-1).
    checks = [qp.prod(*[qp.PauliX(wires=wire) for wire in range(n)])]

    # Neighbouring ZZ parities: Z_i Z_(i+1), i = 0, ..., n-2.
    for i in range(n - 1):
        checks.append(qp.prod(qp.PauliZ(wires=i), qp.PauliZ(wires=i + 1)))

    return checks


def wrapper(n: int) -> float:
    """Build the n-qubit device and average the n parity checks."""
    dev = qp.device("default.qubit", wires=n)

    @qp.qnode(dev)
    def parities():
        return [qp.expval(observable) for observable in prepare_and_build_checks(n)]

    return float(np.mean(np.array(parities())))


# Challenge harness: the checks below audit the actual observables returned.
AUDIT = {"tapes": []}
_real_device = qp.device


def audited_device(*args, **kwargs):
    """Wrap the device so the grader can inspect the measured observables."""
    dev = _real_device(*args, **kwargs)
    inner = dev.execute

    def execute(circuits, *args, **kwargs):
        batch = [circuits] if isinstance(circuits, qp.tape.QuantumScript) else circuits
        for tape in batch:
            AUDIT["tapes"].append(tape)
        return inner(circuits, *args, **kwargs)

    dev.execute = execute
    return dev


def pauli_key(observable):
    """Convert a single Pauli word with coefficient 1 to a canonical string."""
    try:
        sentence = qp.pauli.pauli_sentence(observable)
    except Exception:
        return None
    terms = list(sentence.items())
    if len(terms) != 1:
        return None
    word, coeff = terms[0]
    if not np.isclose(complex(coeff), 1.0):
        return None
    if len(word) == 0:
        return "identity"
    return "".join(str(w) + word[w] for w in sorted(word))


def expected_keys(n):
    keys = ["".join(str(w) + "X" for w in range(n))]
    keys += [str(i) + "Z" + str(i + 1) + "Z" for i in range(n - 1)]
    return sorted(keys)


def violation(n):
    tapes = AUDIT["tapes"]
    if len(tapes) != 1:
        return f"Expected exactly one circuit execution, saw {len(tapes)}."
    measurements = tapes[0].measurements
    for m in measurements:
        if type(m) is not qp.measurements.ExpectationMP:
            return f"Disallowed measurement: {type(m).__name__}."
    if len(measurements) != n:
        return f"You returned {len(measurements)} checks for n = {n}; expected {n}."
    keys = [pauli_key(m.obs) for m in measurements]
    if None in keys:
        return "A check is not a single Pauli word with coefficient 1."
    if "identity" in keys:
        return "A check is the identity."
    if sorted(keys) != expected_keys(n):
        return f"Wrong observables: got {sorted(keys)}, expected {expected_keys(n)}."
    return None


def run(test_case_input: str) -> str:
    n = json.loads(test_case_input)
    AUDIT["tapes"] = []
    qp.device = audited_device
    try:
        value = wrapper(n)
    finally:
        qp.device = _real_device

    message = violation(n)
    if message is not None:
        raise AssertionError(message)
    if not np.isclose(value, 1.0, atol=1e-5):
        raise AssertionError("Not the GHZ state: the checks do not all average to 1.")
    return json.dumps(value)


def check(solution_output: str, expected_output: str) -> None:
    sol = json.loads(solution_output)
    exp = json.loads(expected_output)
    assert np.isclose(sol, exp, atol=1e-5), "Output differs from expected value."


test_cases = [
    ("3", "1.0"),
    ("5", "1.0"),
]


if __name__ == "__main__":
    for i, (input_, expected_output) in enumerate(test_cases):
        print(f"Running test case {i} with input '{input_}'...")
        try:
            output = run(input_)
        except Exception as exc:
            print(f"Runtime Error. {exc}")
        else:
            check(output, expected_output)
            print(f"Correct! Output: {output}")
