
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
import copy

colors = ['#D5B4AB', '#DFE780', '#80E7E1']

# x1 = [0.05, 0.35]
# x2 = [0.1, 0.2]
# x3 = [0.225, 0.275]

x1 = [0, 0.35]
x2 = [0.1, 0.3]
x3 = [0.3, 0.4]


def template():
    N = NN([AffineTransform([[5, 5], [5, -5]]), ReluTransform(), AffineTransform([[5, 10], [5, -5]], bias = [12.5, -25])])

    p1 = N.get_point_output(x1, to_layer = 1)
    p2 = N.get_point_output(x2, to_layer = 1)
    p3 = N.get_point_output(x3, to_layer = 1)

    # print(p1)
    # print(p2)
    # print(p3)

    x3_min = min(p1[0], p2[0], p3[0])
    x3_max = max(p1[0], p2[0], p3[0])
    x4_min = min(p1[1], p2[1], p3[1])
    x4_max = max(p1[1], p2[1], p3[1])

    T_base = [(x3_min, x3_max), (x4_min, x4_max)]

    T_cur = T_base

    fig = plt.figure()
   
    ax = fig.add_subplot(111, aspect='equal')
    ax.set_facecolor('#F6F6F1')

    print('p1:', p1)
    ax.scatter(x=p1[0],y=p1[1], c='r', s=5)
    ax.scatter(x=p2[0],y=p2[1], c='b', s=5)
    ax.scatter(x=p3[0],y=p3[1], c='g', s=5)

    T_curs = []
    T_outs = []

    while(True):
         # add_rectangle(ax, T_cur, fill=False, linestyle='dotted')

         T_out = N.get_output(T_cur, from_layer=1)

         T_curs.append(copy.deepcopy(T_cur))
         T_outs.append(copy.deepcopy(T_out))

         alpha = 2

         print('T_cur:', T_cur)
         print('T_out:', T_out)

         # check property
         if(T_out[0][0] > T_out[1][1]):
             # expand the template
             # x3_mid = (T_cur[0][0] + T_cur[0][1])/2
             # x4_mid = (T_cur[1][0] + T_cur[1][1])/2

             x3_wid = (T_cur[0][1] - T_cur[0][0])
             x4_wid = (T_cur[1][1] - T_cur[1][0])

             T_cur[0] = (T_cur[0][0]-x3_wid/2,T_cur[0][1]+x3_wid/2)
             T_cur[1] = (T_cur[1][0]-x4_wid/2,T_cur[1][1]+x4_wid/2)
         else:
             break

    print(T_cur)

    for TT in T_curs:
    	print('T_curs:', TT)
    	add_rectangle(ax, TT, fill=False, linestyle='dotted')

    add_rectangle(ax, T_curs[-2], fill=False)

    min_x = min(T_cur[0][0]-5, T_cur[1][0]-5)
    max_x = max(T_cur[0][1]+5, T_cur[1][1]+5)

    plt.xlim([min_x, max_x])
    plt.ylim([min_x, max_x])

    print(min_x, max_x)

    fig.savefig('template_im.png', dpi=200, bbox_inches='tight')    

    # Print the output
    fig2 = plt.figure()
   
    ax2 = fig2.add_subplot(111, aspect='equal')
    ax2.set_facecolor('#F6F6F1')

    for TT in T_outs:
    	print(TT)
    	add_rectangle(ax2, TT, fill=False, linestyle='dotted')

    add_rectangle(ax2, T_outs[-2], fill=False)

    ax2.add_artist(lines.Line2D([-500, 500], [-500, 500]))

    min_x = min(T_out[0][0]-5, T_out[1][0]-5)
    max_x = max(T_out[0][1]+5, T_out[1][1]+5)

    plt.xlim([min_x, max_x])
    plt.ylim([min_x, max_x])

    fig2.savefig('template_out.png', dpi=200, bbox_inches='tight')

def graph1():

    T = [(0.5, 3.5), (-2, 1)]

    # T = [(-6.0, 10.0), (-10.875, 9.125)]


    I1 = [(0, 0.1), (0.3, 0.4)]
    I2 = [(0.05, 0.15), (0.15, 0.25)]
    I3 = [(0.2, 0.25), (0.25, 0.30)]

    N = NN([AffineTransform([[5, 5], [5, -5]]), ReluTransform(), AffineTransform([[5, 10], [5, -5]], bias = [12.5, -25])])

    print(N.get_output(I1, to_layer = 1))
    print(N.get_output(I2, to_layer = 1))
    print(N.get_output(I3, to_layer = 1))

    fig = plt.figure()

   
    ax = fig.add_subplot(111, aspect='equal')
    ax.set_facecolor('#F6F6F1')

    # Add template
    add_rectangle(ax, T, fill=False)

    plt.xlim([T[0][0]-1, T[0][1]+1])
    plt.ylim([T[1][0]-1, T[1][1]+1])
    # plt.xlabel('')

    # Add h1(I1)
    add_rectangle(ax, N.get_output(I1, to_layer = 1), fill=True, color=colors[0])

    # Add h1(I1)
    add_rectangle(ax, N.get_output(I2, to_layer = 1), fill=True, color=colors[1])

    # Add h1(I1)
    add_rectangle(ax, N.get_output(I3, to_layer = 1), fill=True, color=colors[2])


    fig.savefig('rect.png', dpi=200, bbox_inches='tight')


    ######################################
    # New figure

    fig2 = plt.figure()

   
    ax2 = fig2.add_subplot(111, aspect='equal')
    ax2.set_facecolor('#F6F6F1')

    T_out = N.get_output(T, from_layer = 1)

    min_x = min(T_out[0][0]-5, T_out[1][0]-5)
    max_x = max(T_out[0][1]+5, T_out[1][1]+5)

    plt.xlim([min_x, max_x])
    plt.ylim([min_x, max_x])

    # Add template
    print("See:")
    print(T)
    print(N.get_output(T, from_layer = 1))
    add_rectangle(ax2, N.get_output(T, from_layer = 1), fill=False)

    # Add h1(I1)
    add_rectangle(ax2, N.get_output(I1), fill=True, color=colors[0])

    # Add h1(I1)
    add_rectangle(ax2, N.get_output(I2), fill=True, color=colors[1])

    # Add h1(I1)
    add_rectangle(ax2, N.get_output(I3), fill=True, color=colors[2])

    ax2.add_artist(lines.Line2D([-500, 500], [-500, 500]))

    fig2.savefig('out_rect.png', dpi=200, bbox_inches='tight')

def graph2():

    T = [(0.5, 3.5), (-2, 1)]

    I1 = [(0, 0.1), (0.3, 0.4)]
    I2 = [(0.05, 0.15), (0.15, 0.25)]
    I3 = [(0.2, 0.25), (0.25, 0.30)]

    # N = NN([AffineTransform([[5, 5], [5, -5]]), ReluTransform(), AffineTransform([[5, 10], [5, -5]])])

    # print(N.get_output(I1, 1))
    # print(N.get_output(I2, 1))
    # print(N.get_output(I3, 1))

    N_app = NN([AffineTransform([[4, 6], [6, -4]]), ReluTransform(), AffineTransform([[6, 9], [4, -6]], bias = [12.5, -25])])

    print(N_app.get_output(I1, to_layer = 1))
    print(N_app.get_output(I2, to_layer = 1))
    print(N_app.get_output(I3, to_layer = 1))

    fig = plt.figure()

   
    ax = fig.add_subplot(111, aspect='equal')
    ax.set_facecolor('#F6F6F1')

    # Add template
    add_rectangle(ax, T, fill=False)

    plt.xlim([T[0][0]-1, T[0][1]+1])
    plt.ylim([T[1][0]-1, T[1][1]+1])
    # plt.xlabel('')

    # Add h1(I1)
    add_rectangle(ax, N_app.get_output(I1, to_layer = 1), fill=True, color=colors[0])

    # Add h1(I1)
    add_rectangle(ax, N_app.get_output(I2, to_layer = 1), fill=True, color=colors[1])

    # Add h1(I1)
    add_rectangle(ax, N_app.get_output(I3, to_layer = 1), fill=True, color=colors[2])


    fig.savefig('rect2.png', dpi=200, bbox_inches='tight')

    ######################################
    # New figure

    fig2 = plt.figure()

   
    ax2 = fig2.add_subplot(111, aspect='equal')
    ax2.set_facecolor('#F6F6F1')

    T_out = N_app.get_output(T, from_layer = 1)

    min_x = min(T_out[0][0]-5, T_out[1][0]-5)
    max_x = max(T_out[0][1]+5, T_out[1][1]+5)

    plt.xlim([min_x, max_x])
    plt.ylim([min_x, max_x])

    # Add template
    print("See:")
    print(T)
    print(N_app.get_output(T, from_layer = 1))
    add_rectangle(ax2, N_app.get_output(T, from_layer = 1), fill=False)

    # Add h1(I1)
    add_rectangle(ax2, N_app.get_output(I1), fill=True, color=colors[0])

    # Add h1(I1)
    add_rectangle(ax2, N_app.get_output(I2), fill=True, color=colors[1])

    # Add h1(I1)
    add_rectangle(ax2, N_app.get_output(I3), fill=True, color=colors[2])

    ax2.add_artist(lines.Line2D([-500, 500], [-500, 500]))

    fig2.savefig('out_rect2.png', dpi=200, bbox_inches='tight')


def graph3():

    T = [(0.5, 3.5), (-2, 1)]

    TM = [(0.5, 3.5), (-2.5, 1.5)]

    I1 = [(0, 0.1), (0.3, 0.4)]
    I2 = [(0.05, 0.15), (0.15, 0.25)]
    I3 = [(0.2, 0.25), (0.25, 0.30)]


    N_app = NN([AffineTransform([[6, 6], [7, -6]]), ReluTransform(), AffineTransform([[6, 9], [4, -6]], bias = [12.5, -25])])

    print(N_app.get_output(I1, to_layer = 1))
    print(N_app.get_output(I2, to_layer = 1))
    print(N_app.get_output(I3, to_layer = 1))

    fig = plt.figure()

   
    ax = fig.add_subplot(111, aspect='equal')
    ax.set_facecolor('#F6F6F1')

    # Add template
    add_rectangle(ax, T, fill=False)

    add_rectangle(ax, TM, fill=False, linestyle='dotted', edgecolor='b')


    plt.xlim([TM[0][0]-1, TM[0][1]+1])
    plt.ylim([TM[1][0]-1, TM[1][1]+1])
    # plt.xlabel('')

    # Add h1(I1)
    add_rectangle(ax, N_app.get_output(I1, to_layer = 1), fill=True, color=colors[0])

    # Add h1(I1)
    add_rectangle(ax, N_app.get_output(I2, to_layer = 1), fill=True, color=colors[1])

    # Add h1(I1)
    add_rectangle(ax, N_app.get_output(I3, to_layer = 1), fill=True, color=colors[2])


    fig.savefig('rect3.png', dpi=200, bbox_inches='tight')

    ######################################
    # New figure

    fig2 = plt.figure()

   
    ax2 = fig2.add_subplot(111, aspect='equal')
    ax2.set_facecolor('#F6F6F1')

    T_out = N_app.get_output(T, from_layer = 1)
    TM_out = N_app.get_output(TM, from_layer = 1)

    min_x = min(TM_out[0][0]-5, TM_out[1][0]-5)
    max_x = max(TM_out[0][1]+5, TM_out[1][1]+5)

    plt.xlim([min_x, max_x])
    plt.ylim([min_x, max_x])

    # Add template
    print("See:")
    print(T)
    print(N_app.get_output(T, from_layer = 1))
    add_rectangle(ax2, N_app.get_output(T, from_layer = 1), fill=False)
    add_rectangle(ax2, N_app.get_output(TM, from_layer = 1), fill=False, edgecolor='b', linestyle='dotted')

    # Add h1(I1)
    add_rectangle(ax2, N_app.get_output(I1), fill=True, color=colors[0])

    # Add h1(I1)
    add_rectangle(ax2, N_app.get_output(I2), fill=True, color=colors[1])

    # Add h1(I1)
    add_rectangle(ax2, N_app.get_output(I3), fill=True, color=colors[2])

    ax2.add_artist(lines.Line2D([-500, 500], [-500, 500]))

    fig2.savefig('out_rect3.png', dpi=200, bbox_inches='tight')

def add_rectangle(ax, shape, color = None, fill=True, edgecolor='g', linestyle = 'solid'):

    leng = shape[0][1] - shape[0][0]
    wid = shape[1][1] - shape[1][0]

    ax.add_patch(
    patches.Rectangle(
        (shape[0][0], shape[1][0]),
        leng,
        wid,
        fill=fill,      # remove background
        edgecolor=edgecolor,
        color = color,
        linestyle=linestyle
    ) ) 

class NN:
    layers = []

    def __init__(self, layers):
        self.layers = layers
        # DO nothing
        return 

    def get_output(self, input, from_layer = 0, to_layer = -1):
        if to_layer == -1:
            to_layer = len(self.layers)

        out = input
        # print(from_layer, to_layer)
        for i in range(from_layer, to_layer):
            out = self.layers[i].apply(out)
        return out

    def get_point_output(self, input, from_layer = 0, to_layer = -1):
        if to_layer == -1:
            to_layer = len(self.layers)

        out = input
        # print(from_layer, to_layer)
        for i in range(from_layer, to_layer):
            out = self.layers[i].apply_point(out)
        return out

class ReluTransform:
    def __init__(self):
        # DO nothing
        return 
    
    def apply(self, input):
        # print(input)
        out = []
        for i in range(len(input)):
            out.append((max(input[i][0], 0), max(input[i][1], 0)))
        return out

    def apply_point(self, input):
        # print(input)
        out = []
        for i in range(len(input)):
            out.append(max(input[i][0], 0))
        return out

class AffineTransform:
    W = []
    def __init__(self, mat, bias=[0, 0]):
        self.W = mat
        self.bias = bias

    def apply(self, input):
        out = []
        for i in range(len(self.W)):
            val1 = 0
            val2 = 0
            for j in range(len(self.W[i])):
                add1 = self.W[i][j]*input[j][0]
                add2 = self.W[i][j]*input[j][1]
                val1 += min(add1, add2)
                val2 += max(add1, add2)
            out.append((min(val1, val2) + self.bias[i], max(val1, val2) + self.bias[i]))

        # print(self.bias, 'bias')
        # print(input)
        # print(out)
        return out

    def apply_point(self, input):
        out = []
        for i in range(len(self.W)):
            val1 = 0
            for j in range(len(self.W[i])):
                add1 = self.W[i][j]*input[j]
                val1 += add1
            out.append(val1 + self.bias[i])

        # print(self.bias, 'bias')
        # print(input)
        # print(out)
        return out

if __name__ == '__main__':
    template()
    graph1()
    graph2()
    graph3()


