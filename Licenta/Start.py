from DQN import Agent
from DbENV import myplotter
from policy_gradient import learnDb, Policy_gradient
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()  #comment for DQN

if __name__ == '__main__':
    myAgent = Policy_gradient()
    #myAgent = Agent()
    try:
        #myAgent.train_db()
        learnDb(myAgent)
    except Exception as e:
        print(str(e))
        myplotter.plotgraph()
