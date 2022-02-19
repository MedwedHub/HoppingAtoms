import numba
import numpy

BOX_RANGE = 1_000_000
PARTICLE_RANGE = 250_000
PARTICLE_SCALE = 3
PARTICLE_COUNT = PARTICLE_SCALE * PARTICLE_SCALE
PARTICLE_RADIUS = 10_000
DIFF_PARTICLE_RADIUS = PARTICLE_RADIUS + 1_500
PARTICLE_RADIUS_SQR = PARTICLE_RADIUS * PARTICLE_RADIUS
DIFF_PARTICLE_RADIUS_SQR = DIFF_PARTICLE_RADIUS * DIFF_PARTICLE_RADIUS
JUMP_RANGE = 3.5
TWO_PI = 6.28319


@numba.jit(numba.float64[::PARTICLE_COUNT, ::2](), nopython=True, nogil=True, cache=True)
def spawn_particle():
    particales = numpy.empty((PARTICLE_COUNT, 2))
    for i in range(PARTICLE_SCALE):
        for j in range(PARTICLE_SCALE):
            particales[i * PARTICLE_SCALE + j] = [PARTICLE_RANGE *
                                                 (i + 1), PARTICLE_RANGE * (j + 1)]
    return particales

@numba.jit(numba.float64[:, ::2](), nopython=True, nogil=True, parallel=True, cache=True)
def spawn_atoms_1kk():
    return BOX_RANGE * numpy.random.random((1_000_00, 2))

@numba.jit(numba.float64[:, ::2](numba.int32), nopython=True, nogil=True, parallel=True, cache=True)
def generate_jump(count):
    angle = TWO_PI * numpy.random.random(count)
    jump_x = JUMP_RANGE * numpy.cos(angle)
    jump_y = JUMP_RANGE * numpy.sin(angle)
    return numpy.column_stack((jump_x, jump_y))

@numba.jit(numba.float64[:, ::2](numba.float64[:, ::2], numba.float64[::PARTICLE_COUNT, ::2]), nopython=True, nogil=True, cache=True)
def jumping(atoms: numpy.ndarray, particle: numpy.ndarray):
    size = atoms.shape[0]
    generate_count = 1_000
    for i in range(1_000_000):
        if i // generate_count == 0:
            jump = generate_jump(generate_count * size)
            slice_ind = 0
        atoms += jump[slice_ind * size : (slice_ind + 1) * size - 1,:]
        slice_ind += 1
    return atoms

@numba.jit(numba.float64[:, ::2](numba.float64[:, ::2], numba.float64[::PARTICLE_COUNT, ::2]), nopython=True, nogil=True, parallel=True, cache=True)
def filter_atoms_at_spawn(atoms: numpy.ndarray, particle: numpy.ndarray):
    atoms_return = numpy.empty((0, 2))
    for i in range(PARTICLE_COUNT):
        x_diff = atoms[:,0] - particle[i, 0]
        y_diff = atoms[:,1] - particle[i, 1]
        dist_sqr = x_diff * x_diff + y_diff * y_diff
        dist_filter = (dist_sqr <= DIFF_PARTICLE_RADIUS_SQR) & (dist_sqr > PARTICLE_RADIUS_SQR)
        atoms_return = numpy.concatenate((atoms_return, atoms[dist_filter]), 0)
    return atoms_return

@numba.jit(numba.float64[:, ::2](numba.float64[::PARTICLE_COUNT, ::2]), nopython=True, nogil=True, cache=True)
def spawn(particales: numpy.ndarray):
    atoms_return = numpy.empty((0, 2))
    for i in range(1_000):
        atoms = spawn_atoms_1kk()
        atoms_return = numpy.concatenate((atoms_return, filter_atoms_at_spawn(atoms, particales)), 0)
    return atoms_return

particales = spawn_particle()
atoms_spawned = spawn(particales)

print(atoms_spawned.shape[0])

jumping(atoms_spawned, particales)
print('Готово, блять')
