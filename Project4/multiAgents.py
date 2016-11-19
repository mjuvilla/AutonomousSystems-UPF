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
from searchAgents import mazeDistance

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
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

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

        if not food_position:
            return float("inf")

        # list that contains the distance between the remaining food and the agent
        diff_food_agent = [(abs(newPos[0] - x) + abs(newPos[1] - y)) for x, y in food_position]
        # list that contains the distance between the ghosts and the agent
        diff_ghosts_agent = [
            abs(newPos[0] - ghostState.configuration.pos[0]) + abs(newPos[1] - ghostState.configuration.pos[1]) for
            ghostState in newGhostStates]

        # assign more score to positions closer to food
        closest_food_score = 10 / float(min(diff_food_agent))

        ghosts_score = 0
        # if the distance to a ghost is zero, this move is a really bad idea
        # also, if a ghost is at distance 1, it is not a good idea to stop
        if 0.0 in diff_ghosts_agent or (1.0 in diff_ghosts_agent and action == "Stop"):
            ghosts_score = - float("inf")

        # we also take into account the length of the remaining dot foods, as if this movement makes the agent eat
        # a food dot, we will have an element less in this list
        return closest_food_score + ghosts_score - 10 * len(diff_food_agent)


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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def minimax(self, state, depth, agent, num_agents):
        # if we have completed a turn, reduce the depth in 1
        if agent % num_agents == 0 and agent != 0:
            depth -= 1

        agent = agent % num_agents
        legal_actions = state.getLegalActions(agent)

        # if it is a terminal node (no more moves to do), return the score
        if depth == 0 or not legal_actions:
            return self.evaluationFunction(state), None

        if agent == 0:
            scores = [(self.minimax(state.generateSuccessor(agent, action), depth, agent + 1, num_agents)[0], action)
                      for action in legal_actions]
            return sorted(scores, key=lambda x: x[0], reverse=True)[0]

        else:
            scores = [(self.minimax(state.generateSuccessor(agent, action), depth, agent + 1, num_agents)[0], action)
                      for action in legal_actions]
            return sorted(scores, key=lambda x: x[0])[0]

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
        return self.minimax(gameState, self.depth, 0, gameState.getNumAgents())[1]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def alphabeta(self, state, depth, alpha, beta, agent, num_agents):
        if agent % num_agents == 0 and agent != 0:
            depth -= 1

        agent = agent % num_agents
        legal_actions = state.getLegalActions(agent)

        if depth == 0 or not legal_actions:
            return self.evaluationFunction(state), None

        bestAction = None

        if agent == 0:
            bestValue = float("-inf")

            for action in legal_actions:
                v = self.alphabeta(state.generateSuccessor(agent, action), depth, alpha, beta, agent + 1, num_agents)[0]
                if v > bestValue:
                    bestValue = v
                    bestAction = action
                if beta < bestValue:
                    break
                alpha = max(alpha, bestValue)
            return bestValue, bestAction

        else:
            bestValue = float("inf")
            for action in legal_actions:
                v = self.alphabeta(state.generateSuccessor(agent, action), depth, alpha, beta, agent + 1, num_agents)[0]
                if v < bestValue:
                    bestValue = v
                    bestAction = action
                if bestValue < alpha:
                    break
                beta = min(beta, bestValue)
            return bestValue, bestAction

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        alpha = float("-inf")
        beta = float("inf")
        return self.alphabeta(gameState, self.depth, alpha, beta, 0, gameState.getNumAgents())[1]

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def expectimax(self, state, depth, agent, num_agents):
        if agent % num_agents == 0 and agent != 0:
            depth -= 1

        agent = agent % num_agents
        legal_actions = state.getLegalActions(agent)

        if depth == 0 or not legal_actions:
            return self.evaluationFunction(state), None

        if agent == 0:
            scores = [(self.expectimax(state.generateSuccessor(agent, action), depth, agent + 1, num_agents)[0], action)
                      for action in legal_actions]

            #it improves the results, but not sure if we can do this

            # stop_action = False
            # stop_score = 0
            # stop_index = 0
            # for idx, (score, action) in enumerate(scores):
            #     if action == "Stop":
            #         stop_score = score
            #         stop_index = idx
            #         stop_action = True
            #
            # if stop_action:
            #     for score, action in scores:
            #         if action != "Stop":
            #             if score == stop_score:
            #                 scores[stop_index] = (-float("inf"), "Stop")

            return sorted(scores, key=lambda x: x[0], reverse=True)[0]

        else:
            scores = [(self.expectimax(state.generateSuccessor(agent, action), depth, agent + 1, num_agents)[0], action)
                      for action in legal_actions]

            if float("inf") in zip(*scores)[0] and -float("inf") in zip(*scores)[0]:
                return -float("inf"), None

            return sum(zip(*scores)[0]) / len(scores), None

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        return self.expectimax(gameState, self.depth, 0, gameState.getNumAgents())[1]

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    if currentGameState.isWin():
        return float("inf")
    if currentGameState.isLose():
        return - float("inf")

    score = 0

    newPos = currentGameState.getPacmanPosition()

    newFood = currentGameState.getFood()
    newCapsules = currentGameState.getCapsules()

    # get the position of each food dot
    food_position = []
    for row, food_rows in enumerate(newFood):
        for col, food in enumerate(food_rows):
            if food:
                food_position.append((row, col))

    # list that contains the distance between the remaining food and the agent
    diff_food_agent = [util.manhattanDistance(newPos, food) for food in
                       food_position]  # list that contains the distance between the ghosts and the agent

    diff_capsules_agent = [util.manhattanDistance(newPos, capsule) for
                         capsule in newCapsules]

    closest_food = sorted(food_position, key=lambda x: util.manhattanDistance(newPos, x))[:3]
    closest_capsules = sorted(newCapsules, key=lambda x: util.manhattanDistance(newPos, x))[:3]

    food_left = len(diff_food_agent)
    capsules_left = len(diff_capsules_agent)

    diff_food_agent = [mazeDistance(newPos, food, currentGameState) for food in closest_food]
    diff_capsules_agent = [mazeDistance(newPos, capsule, currentGameState) for capsule in closest_capsules]

    if capsules_left:
        score += - capsules_left + 1 / float(min(diff_capsules_agent))
    score += -0.03 * food_left + 1 / float(min(diff_food_agent)) + currentGameState.getScore()

    return score

def square(x):
    return x*x

better = betterEvaluationFunction
