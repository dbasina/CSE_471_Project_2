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
    newPos = successorGameState.getPacmanPosition()
    newFood = successorGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    ghostPosition = successorGameState.getGhostPosition(1)
    capsulePositionList = successorGameState.getCapsules();
    ghostDistance = math.sqrt(math.pow(newPos[1]-ghostPosition[1],2) + math.pow(newPos[0]-ghostPosition[0],2));

    # Go eat the capsule first
    if (len(capsulePositionList)!=0):
        capsulePositions = capsulePositionList[0]

        print "CapsulePositions:", capsulePositions
        capsuleDistance = math.sqrt(math.pow(newPos[1]-capsulePositions[1],2) + math.pow(newPos[0]-capsulePositions[0],2));
        newScore = 0.40*successorGameState.getScore() + 0.30*ghostDistance - 0.30*capsuleDistance ;
        print "Pacman :",newPos," Ghost: ",ghostPosition, "Distance: ",ghostDistance, "GameStatescore: ",successorGameState.getScore(), "New Score: ",newScore
        #print newScaredTimes
        return newScore
    else:

        # Chase ghost if ghost is scared
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        print "Scared Times: ",newScaredTimes
        if (newScaredTimes[0] > 0):
            newScore = 0.60*successorGameState.getScore() - 0.40*ghostDistance;

        # Eat remaining food if not scared.
        else:
            newScore = 0.30*successorGameState.getScore() + 0.20*ghostDistance-0.45*successorGameState.getNumFood();


        return newScore


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
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        """
        numberOfAgents = gameState.getNumAgents();
    agentCycle = []
    for i in range(self.depth):
        for j in range(numberOfAgents):
            agentCycle.append(j);
    print agentCycle
    agentTracker=0
    depth = 0
    maxDepth = len(agentCycle)-1
    maxDepthReached = False
    state = gameState

    def recursiveMax(depth,state,agentTracker,agentCycle,maxDepthReached):
        print "\nrecursiveMax: "
        print "depth: ",depth
        print "agentTracker: ",agentTracker
        print "agentCycle[agentTracker] :",agentCycle[agentTracker]

        if (depth == maxDepth):
            maxDepthReached = True;
            print "Utility:",scoreEvaluationFunction(state)
            return scoreEvaluationFunction(state)

        else:
            successors = []
            legalActions = state.getLegalActions(agentCycle[agentTracker]);
            print "actions:",legalActions
            for i in legalActions:
                successors.append(state.generateSuccessor(agentCycle[agentTracker],i));

            depth = depth + 1
            agentTracker=agentTracker+1;


            successorUtilities = [];
            for i in successors:
                successorUtilities.append(recursiveMin(depth,i,agentTracker,agentCycle,maxDepthReached))
            print successorUtilities
            if (maxDepthReached==True):
                maxValue = max(successorUtilities)
                actionIndex = successorUtilities.index(maxValue);
                action = legalActions(actionIndex);
                return maxValue

    def recursiveMin(depth,state,agentTracker,agentCycle,maxDepthReached):
        print "\nrecursiveMin: "
        print "depth: ",depth
        print "agentTracker: ",agentTracker
        print "agentCycle[agentTracker] :",agentCycle[agentTracker]

        if (depth == maxDepth):
            maxDepthReached = True;
            print "Utility:",scoreEvaluationFunction(state)
            return scoreEvaluationFunction(state)

        else:
            successors = []
            legalActions = state.getLegalActions(agentCycle[agentTracker]);
            print "actions:",legalActions
            for i in legalActions:
                successors.append(state.generateSuccessor(agentCycle[agentTracker],i));

            depth = depth + 1
            agentTracker=agentTracker+1;

            successorUtilities = [];
            for i in successors:
                if (agentCycle[agentTracker]!= 0):
                    successorUtilities.append(recursiveMin(depth,i,agentTracker,agentCycle,maxDepthReached))
                else:
                    successorUtilities.append(recursiveMax(depth,i,agentTracker,agentCycle,maxDepthReached))
            print successorUtilities
            if (maxDepthReached==True):
                minValue = min(successorUtilities)
                actionIndex = successorUtilities.index(minValue);
                action = legalActions(actionIndex);
                return minValue

    print recursiveMax(0,state,agentTracker,agentCycle,maxDepthReached)
    """
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

    max_value(gameState, game_depth, pacman)
        """

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

        max_value(gameState, game_depth, pacman)

        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    pacman = 0
    ghosts = 1
    state = gameState
    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        pacman = 0

        #unfinished
        """
        def max_value(state,depth, alpha, beta,agent):
            if state:  #if state is terminal
                print "return utility(state)"
            best_value = float("-inf")
            actions = state.getLegalActions(agent)
            print "define best_action default somehow"
            for a in actions:
                value = min_value(state.generateSuccessor(agent, a))
                best_value = max(best_value, value)
                best_action = a
                if best_value >= beta:
                    return best_action
                alpha = max(alpha, best_value)
            return best_action

        def min_value(state, alpha, beta):
            if state:  # if terminal
                print "return utility(state)"
            ghost = ghost + 1
            best_value = float("inf")
            actions = state.getLegalActions(pacman)
            for a in actions:
                if ghost != 5:  # bc there's 4 ghosts, so iterate 4 times?
                    value = minValue(state.generateSuccessor(ghost, a))
                    best_value = min(best_value, value)
                else:  # if on the last ghost
                    print "do something"
            if best_value <= alpha:
                return best_value
            beta = min(beta, best_value)
            return best_value

        max_value(gameState,depth,alpha,beta,agent)
        """
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
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
