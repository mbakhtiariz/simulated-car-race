from __future__ import print_function
import neat
import numpy as np
from pytocl.main import main
from my_driver import MyDriver
from sklearn.externals import joblib
from neat import Config


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = 0
        net = neat.nn.RecurrentNetwork.create(genome, config)
        joblib.dump(net, path+"neat_model_%d.pkl"%exp_num)
        main(MyDriver())

        fitness_file = open(path+"fitness_%d.txt"%exp_num, 'r')
        for f in fitness_file:
            print("f:",f)
            genome.fitness = float(f)
        print("genome.fitness:",genome.fitness)


path = 'neatNetworks/'
exp_num = 8
# Load configuration.
config = Config(neat.DefaultGenome, neat.DefaultReproduction,
                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                path+'config_'+str(exp_num))

# Create the population, which is the top-level object for a NEAT run.
p = neat.Population(config)

# Add a stdout reporter to show progress in the terminal.
p.add_reporter(neat.StdOutReporter(False))
p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)

# Run until a solution is found.
winner = p.run(eval_genomes, 60)

# Display the winning genome.
print('\nBest genome:\n{!s}'.format(winner))

# Show output of the most fit genome against training data.
print('\nOutput:')
winner_net = neat.nn.RecurrentNetwork.create(winner, config)
joblib.dump(winner_net, path+'winner_net_%d.pkl'%exp_num)
node_names = {}
node_names[0]= 'Steer'

