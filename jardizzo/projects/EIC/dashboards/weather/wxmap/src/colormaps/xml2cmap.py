import os
import sys

file = sys.argv[1]
name, ext = os.path.splitext(file)
name = os.path.basename(name)

stop = []
red = []
green = []
blue = []

with open(file, 'r') as f:
    lines = f.readlines()

for line in lines:

    line = line.strip()
    if 'stop position' in line:
        line = line.split('"')
        stop.append(float(line[1]))
        red.append(float(line[3])/255)
        green.append(float(line[5])/255)
        blue.append(float(line[7])/255)

#print('attribute:')
#print(' '*2 + 'colorbar:')
print(' '*4 + name + ':')

print(' '*6 + 'red:')
for i,v in enumerate(red):
    print(' '*8 + f'- {stop[i]:.3f} {v:.3f} {v:.3f}')

print(' '*6 + 'green:')
for i,v in enumerate(green):
    print(' '*8 + f'- {stop[i]:.3f} {v:.3f} {v:.3f}')

print(' '*6 + 'blue:')
for i,v in enumerate(blue):
    print(' '*8 + f'- {stop[i]:.3f} {v:.3f} {v:.3f}')

print()
