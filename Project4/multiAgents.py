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
import random, util

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
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # get the position of each food dot
        food_position = []
        for row, food_rows in enumerate(newFood):
            for col, food in enumerate(food_rows):
                if food:
                    food_position.append((row, col))

        # list that contains the distance between the remaining food and the agent
        diff_food_agent = [(abs(newPos[0]-x)+abs(newPos[1]-y)) for x, y in food_position]
        # list that contains the distance between the ghosts and the agent
        diff_ghosts_agent = [abs(newPos[0]-ghostState.configuration.pos[0]) + abs(newPos[1]-ghostState.configuration.pos[1]) for ghostState in newGhostStates]

        # assign more score to positions closer to food
        if diff_food_agent: # this checks if diff_food_agent is not empty
            closest_food_score = 10/float(min(diff_food_agent))
        # if the list is empty, it means that with this move we win
        else:
            closest_food_score = 10

        ghosts_score = 0
        # if the distance to a ghost is zero, this move is a really bad idea
        # also, if a ghost is at distance 1, it is not a good idea to stop
        if 0.0 in diff_ghosts_agent or (1.0 in diff_ghosts_agent and action == "Stop"):
            ghosts_score = -1000

        # we also take into account the length of the remaining dot foods, as if this movement makes the agent eat
        # a food dot, we will have an element less in this list
        return closest_food_score + ghosts_score - 10*len(diff_food_agent)

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

    def minimax(self, state, depth, agent, num_agents):
        if agent % num_agents == 0 and agent != 0:
            depth -= 1

        if depth == 0:
            return self.evaluationFunction(state)

        agent = agent % num_agents
        legal_actions = state.getLegalActions(agent)

        if not state.getLegalActions(agent):
            return self.evaluationFunction(state)

        if agent == 0:
            bestValue = float("-inf")
            for action in legal_actions:
                future_state = state.generateSuccessor(agent, action)
                v = self.minimax(future_state, depth, agent+1, num_agents)
                bestValue = max(bestValue, v)
            return bestValue

        else:
            bestValue = float("inf")
            for action in legal_actions:
                future_state = state.generateSuccessor(agent, action)
                v = self.minimax(future_state, depth, agent + 1, num_agents)
                bestValue = min(bestValue, v)
            return bestValue


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
        legal_actions = gameState.getLegalActions(0)
        score_actions = dict()
        for action in legal_actions:
            future_state = gameState.generateSuccessor(0, action)
            score_actions[action] = self.minimax(future_state, self.depth, 1, gameState.getNumAgents())

        return max(score_actions, key=score_actions.get)

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
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
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

