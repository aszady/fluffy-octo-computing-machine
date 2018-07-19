from pyquil.api import CompilerConnection, get_devices
from pyquil.gates import X, H, CNOT, MEASURE
from pyquil.quil import Program

program = Program(
    # put your program here...
    X(0),
)

agave = get_devices(as_dict=True)['8Q-Agave']
compiler = CompilerConnection(device=agave)
compiled = compiler.compile(program)

print(compiled)
