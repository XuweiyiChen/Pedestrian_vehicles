import tensorflow as tf
print('Loaded TF Version:', tf.__version__,'\n')

'''
TF四则运算的实战
'''

def basic_operation():
    V1 = tf.Variable(8)
    V2 = tf.Variable(4)
    addv = V1 + V2
    print(addv)

    sess = tf.Session()
    tf.global_variables_initializer().run(session=sess)
    print('V1 + V2 = ', addv.eval(session=sess))
#
    graph = tf.Graph()
    with graph.as_default():
        a = tf.constant([5,8])
        b = tf.constant([2,4])
        mul = a * b
    with tf.Session(graph=graph) as mySess:
        tf.global_variables_initializer().run()
        print('a * b = ',mySess.run(mul))
    #
#     #
    #Placeholder
    graph = tf.Graph()
    with graph.as_default():
        a = tf.placeholder(dtype=tf.float32)
        b = tf.Variable([2,3],dtype=tf.float32)
        mul = a * b

    with tf.Session(graph=graph) as mySess:
        tf.global_variables_initializer().run()
        value = load_from_remote()
#
        for partialvalue in load_partial(value,2):
            runResult = mySess.run(mul,feed_dict={a:partialvalue})
            print('a * b = ', runResult)
#
#
#
def load_from_remote():
    return [-x for x in range(1000)]   # 0, -1, -2, -3, ...., -999
#
def load_partial(value, step):
    index = 0
    while index < len(value):
        yield value[index:index+step]
        index +=step
    return
#
# '''
# 第一次取出的值是多少？ P1 = value[0:2] = [0,-1]   b = [2,3]   P1 * b =[0,-3]
# 第二次              p2 = value[2:4] = [-2,-3]   b = [2,3]   P2 * b =[-4,-9]
# ......
#
# '''





if __name__ == "__main__":
    basic_operation()