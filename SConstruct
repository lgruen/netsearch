import os, glob

target = 'diginets'

sources = Split(glob.glob('src/*.cpp'))

env = Environment(PATH = os.environ['PATH'])

# pick one:
env.Append(CXXFLAGS='-Wall -W -pedantic -Wno-long-long -g')
# env.Append(CXXFLAGS='-Wall -W -pedantic -Wno-long-long -O3 -march=athlon-xp')
# env.Append(CXXFLAGS='-Wall -W -pedantic -Wno-long-long -fopenmp -O3 -g -msse2')

# needed with -fopenmp:
# env.Append(LINKFLAGS='-lgomp')

# used by colorgcc
env.Append(ENV = {'PATH' : os.environ['PATH'],
                     'TERM' : os.environ['TERM'],
                     'HOME' : os.environ['HOME']})

if 'CXXFLAGS' in os.environ:
	env.Append(CXXFLAGS = os.environ['CXXFLAGS'])

env.Program(target, sources, CPPPATH=['./include/', '.'])

