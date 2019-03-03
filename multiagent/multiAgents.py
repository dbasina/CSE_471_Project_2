# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util,math

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newGhostStates = successorGameState.getGhostStates()
        ghostPosition = successorGameState.getGhostPosition(1)
        legalActions = successorGameState.getLegalActions()

        currentPos = successorGameState .getPacmanPosition()
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        foodList = newFood.asList();

        foodPenalty=0
        deathPenalty=1500
        stillPenalty = 100;
        for i in foodList:
            xpos = i[0]
            ypos = i[1]
            if (newFood[xpos][ypos] == True):
                foodPenalty = foodPenalty+1
        newScore = 10000 - foodPenalty
        if (newPos==successorGameState.getPacmanPosition):
            newScore = 10000-stillPenalty
        # Check if ghost can come close to us
        for i in legalActions:
            possibleState = successorGameState.generatePacmanSuccessor(i)

            if (possibleState.getGhostPosition(1)==newPos):
                newScore = -1000

        return newScore

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    def getAction(self, gameState):

        # Variables
        numberOfAgents = gameState.getNumAgents();
        agentCycle = []
        for i in range(self.depth):
            for j in range(numberOfAgents):
                agentCycle.append(j);
        agentCycle.append(0);
        #print agentCycle
        #print gameState.isWin();
        #print gameState.isLose();
        #print "Max Depth = ",self.depth
        agentTracker=0
        depth = 0
        maxDepth = len(agentCycle)-1
        #maxDepthReached = False
        state = gameState
        actions=gameState.getLegalActions()

        # Variables Used in Recursion
        # depth
        # state
        # agentTracker
        # agentCycle

        def recursiveMax(depth,state,agentTracker,agentCycle):
            #print "\nrecursiveMax: "
            #print "depth: ",depth
            #print "agentTracker: ",agentTracker
            #print "agentCycle[agentTracker] :",agentCycle[agentTracker]

            if (depth == maxDepth):
                return scoreEvaluationFunction(state)


            else:
                successors = []
                legalActions = state.getLegalActions(agentCycle[agentTracker]);
                if (len(legalActions)>0):
                    #print "actions:",legalActions
                    for i in legalActions:
                        successors.append(state.generateSuccessor(agentCycle[agentTracker],i));

                    # Setup Variable to make recursive call
                    nextDepth = depth + 1
                    nextAgentTracker=agentTracker+1;
                    successorUtilities = [];
                    if (nextAgentTracker<len(agentCycle)):
                        for i in successors:
                            successorUtilities.append(recursiveMin(nextDepth,i,nextAgentTracker,agentCycle))

                    else:
                        print "ERROR TRYING TO EXPAND TERMINAL STATE"

                    #print "\nBACK FROM RECURSION AT STATE:"
                    #print "recursiveMin: "
                    #print "depth: ",depth
                    #print "agentTracker: ",agentTracker
                    #print "agentCycle[agentTracker] :",agentCycle[agentTracker]
                    #print "Successor Utilities",successorUtilities
                    #print "\n"
                    maxValue = max(successorUtilities)
                    actionIndex = successorUtilities.index(maxValue);
                    action = legalActions[actionIndex]
                    return (maxValue,action)
                else:
                    #print "State has no legal Actions"
                    return scoreEvaluationFunction(state)

        def recursiveMin(depth,state,agentTracker,agentCycle):
            #print "\nrecursiveMin: "
            #print "depth: ",depth
            #print "agentTracker: ",agentTracker
            #print "agentCycle[agentTracker] :",agentCycle[agentTracker]

            if (depth == maxDepth):
                return scoreEvaluationFunction(state)

            else:
                successors = []
                legalActions = state.getLegalActions(agentCycle[agentTracker]);
                if (len(legalActions)>0):
                    #print "actions:",legalActions

                    for i in legalActions:
                        successors.append(state.generateSuccessor(agentCycle[agentTracker],i));

                    # Setup Variables to make recursive call
                    nextDepth = depth + 1
                    nextAgentTracker=agentTracker+1;
                    successorUtilities = [];
                    if (nextAgentTracker<len(agentCycle)):
                        for i in successors:
                            if (agentCycle[nextAgentTracker]!= 0):
                                successorUtilities.append(recursiveMin(nextDepth,i,nextAgentTracker,agentCycle))
                            else:
                                successorUtilities.append(recursiveMax(nextDepth,i,nextAgentTracker,agentCycle))
                    else:
                        print "ERROR TRYING TO EXPAND TERMINAL STATE";

                    #print "\nBACK FROM RECURSION AT STATE:"
                    #print "recursiveMin: "
                    #print "depth: ",depth
                    #print "agentTracker: ",agentTracker
                    #print "agentCycle[agentTracker] :",agentCycle[agentTracker]
                    #print "Successor Utilities",successorUtilities
                    #print "\n"

                    minValue = min(successorUtilities)
                    actionIndex = successorUtilities.index(minValue);
                    action = legalActions[actionIndex];
                    return (minValue,action)
                else:
                    #print "State has no legal Actions"
                    return scoreEvaluationFunction(state)

        a = recursiveMax(0,state,agentTracker,agentCycle)
        print a[1]
        return a[1]
        util.raiseNotDefined()
    pacman = 0
    ghost = 1

    def max_value(state,depth,agent):
        if state:  #if state is terminal
            print "return utility(state)"
        print "define best_action default somehow"
        best_value = float("-inf")
        actions = state.getLegalActions(agent)
        for a in actions:
            value = min_value(state.generateSuccessor(agent, a),depth,ghost)
            best_value = max(best_value, value)
            best_action = a
        return best_action

    def min_value(state,depth,agent):
        if state: #if state is terminal
            print "return utility(state)"
        print "define best_action default somehow"
        ghost_agent = agent
        agent = agent + 1
        if agent == state.getNumAgents() - 1: #num of ghosts (-1 for pacman)
            print "return state value"
        else:
            best_value = float("inf")
            actions = state.getLegalActions(ghost)
            for a in actions:
                value = min_value(state.generateSuccessor(ghost, a),depth,ghost_agent)
                best_value = min(best_value, value)
                best_action = a
            return best_action

    #max_value(gameState, game_depth, pacman)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # Variables
        numberOfAgents = gameState.getNumAgents()
        agentTracker = 0
        maxDepth = self.depth

        # Variables Used in Recursion
        # depth
        # state
        # agentTracker
        # agentCycle

        def recursiveMax(depth, state, agentTracker, alpha, beta):
            legalActions = state.getLegalActions(agentTracker)
            if state.isWin() or state.isLose():
                return scoreEvaluationFunction(state)
            # Setup Variable to make recursive call
            nextAgentTracker = agentTracker + 1
            maxValue = float("-inf")
            for i in legalActions:
                value = recursiveMin(depth, state.generateSuccessor(agentTracker, i), nextAgentTracker, alpha, beta)
                if value > maxValue:
                    maxValue = value
                    action = i

                if maxValue > beta:
                    return maxValue
                alpha = max(alpha, maxValue)
            if depth == 0:
                return action

            return maxValue


        def recursiveMin(depth, state, agentTracker, alpha, beta):
            legalActions = state.getLegalActions(agentTracker)
            if state.isWin() or state.isLose():
                # print "State has no legal Actions"
                return scoreEvaluationFunction(state)
            # Setup Variables to make recursive call
            nextAgentTracker = (agentTracker + 1) % numberOfAgents
            minValue = float("inf")
            for i in legalActions:
                if nextAgentTracker != 0:
                    value = recursiveMin(depth, state.generateSuccessor(agentTracker, i), nextAgentTracker, alpha, beta)
                else:
                    if depth == maxDepth - 1:
                        value = scoreEvaluationFunction(state.generateSuccessor(agentTracker, i))
                    else:
                        value = recursiveMax(depth + 1, state.generateSuccessor(agentTracker, i), nextAgentTracker, alpha, beta)
                minValue = min(minValue, value)
                if minValue < alpha:
                    return minValue
                beta = min(beta, minValue)
            return minValue

        a = recursiveMax(0, gameState, agentTracker, float("-inf"), float("inf"))
        return a

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):

        # Variables
        numberOfAgents = gameState.getNumAgents();
        agentCycle = []
        for i in range(self.depth):
            for j in range(numberOfAgents):
                agentCycle.append(j);
        agentCycle.append(0);
        #print agentCycle
        #print gameState.isWin();
        #print gameState.isLose();
        #print "Max Depth = ",self.depth
        agentTracker=0
        depth = 0
        maxDepth = len(agentCycle)-1
        #maxDepthReached = False
        state = gameState
        actions=gameState.getLegalActions()
        # Variables Used in Recursion
        # depth
        # state
        # agentTracker
        # agentCycle

        def recursiveMax(depth,state,agentTracker,agentCycle):
            #print "\nrecursiveMax: "
            #print "depth: ",depth
            #print "agentTracker: ",agentTracker
            #print "agentCycle[agentTracker] :",agentCycle[agentTracker]

            if (depth == maxDepth):
                return (scoreEvaluationFunction(state),'NONE')


            else:
                successors = []
                legalActions = state.getLegalActions(agentCycle[agentTracker]);
                if (len(legalActions)>0):
                    #print "actions:",legalActions
                    for i in legalActions:
                        successors.append(state.generateSuccessor(agentCycle[agentTracker],i));

                    # Setup Variable to make recursive call
                    nextDepth = depth + 1
                    nextAgentTracker=agentTracker+1;
                    successorUtilities = [];
                    if (nextAgentTracker<len(agentCycle)):
                        for i in successors:
                            successorUtilities.append(recursiveMin(nextDepth,i,nextAgentTracker,agentCycle))

                    else:
                        print "ERROR TRYING TO EXPAND TERMINAL STATE"

                    #print "\nBACK FROM RECURSION AT STATE:"
                    #print "recursiveMin: "
                    #print "depth: ",depth
                    #print "agentTracker: ",agentTracker
                    #print "agentCycle[agentTracker] :",agentCycle[agentTracker]
                    #print "Successor Utilities",successorUtilities
                    #print "\n"
                    weightedUtilities=[]
                    for i in range(len(legalActions)):
                        util = successorUtilities[i]
                        weightUtil = util[0] * (1.0/float(len(legalActions)))
                        weightedUtilities.append(weightUtil)

                    maxValue = max(weightedUtilities)
                    actionIndex = weightedUtilities.index(maxValue);
                    action = legalActions[actionIndex]
                    return (maxValue,action)
                else:
                    #print "State has no legal Actions"
                    return (scoreEvaluationFunction(state), 'NONE')

        def recursiveMin(depth,state,agentTracker,agentCycle):
            #print "\nrecursiveMin: "
            #print "depth: ",depth
            #print "agentTracker: ",agentTracker
            #print "agentCycle[agentTracker] :",agentCycle[agentTracker]

            if (depth == maxDepth):
                return (scoreEvaluationFunction(state),'NONE')

            else:
                successors = []
                legalActions = state.getLegalActions(agentCycle[agentTracker]);
                if (len(legalActions)>0):
                    #print "actions:",legalActions

                    for i in legalActions:
                        successors.append(state.generateSuccessor(agentCycle[agentTracker],i));

                    # Setup Variables to make recursive call
                    nextDepth = depth + 1
                    nextAgentTracker=agentTracker+1;
                    successorUtilities = [];
                    if (nextAgentTracker<len(agentCycle)):
                        for i in successors:
                            if (agentCycle[nextAgentTracker]!= 0):
                                successorUtilities.append(recursiveMin(nextDepth,i,nextAgentTracker,agentCycle))
                            else:
                                successorUtilities.append(recursiveMax(nextDepth,i,nextAgentTracker,agentCycle))
                    else:
                        print "ERROR TRYING TO EXPAND TERMINAL STATE";

                    #print "\nBACK FROM RECURSION AT STATE:"
                    #print "recursiveMin: "
                    #print "depth: ",depth
                    #print "agentTracker: ",agentTracker
                    #print "agentCycle[agentTracker] :",agentCycle[agentTracker]
                    #print "Successor Utilities",successorUtilities
                    #print "\n"
                    weightedUtilities=[]
                    for i in range(len(legalActions)):
                        util = successorUtilities[i]
                        weightUtil = util[0] * (1.0/float(len(legalActions)))
                        weightedUtilities.append(weightUtil)

                    minValue = min(weightedUtilities)
                    actionIndex = weightedUtilities.index(minValue);
                    action = legalActions[actionIndex];
                    return (minValue,action)
                else:
                    #print "State has no legal Actions"
                    return (scoreEvaluationFunction(state),'NONE')

        a = recursiveMax(0,state,agentTracker,agentCycle)
        print a[1]
        return a[1]
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    #Don't need to implement this
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
