import matplotlib.pyplot as plt

from k_road.model.tire_model.magic_tire_model_93 import MagicTireModel93

m = MagicTireModel93()
a = [2 * pi / 3 * (i / 1000) - (pi / 3) for i in range(1000)]
Vc = 5

for k in [0, .05, .1, .2, .4, .8, 1.6]:
    plt.plot([a * 180 / pi for a in a], [m.calc_all_values(4000, Vc, -a, k, 0)[1] for a in a])
plt.show()

k = [2 * (i / 1000) - 1 for i in range(1000)]
for ad in [0, 2, 5, 10, 20, 40, 80]:
    plt.plot(k, [m.calc_all_values(4000, Vc, -ad * pi / 180, k, 0)[1] for k in k])
plt.show()

# = [m.get_longitudinal_force(4000, 10*cos(a), 10*sin(a), -.01*10*cos(a), 0)[2] for a in x]
