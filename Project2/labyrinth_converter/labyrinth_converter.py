import argparse
import sys
import os.path

def parse_labyrinth(labyrinth_filename):
    with open(labyrinth_filename, "r") as labyrinth_file:
        labyrinth = labyrinth_file.read()
    labyrinth = labyrinth[labyrinth.find("%"):labyrinth.rfind("%")+1].rstrip()
    labyrinth_list = labyrinth.split("\n")[1:-1]
    labyrinth_list = [row[1:-1] for row in labyrinth_list]

    blank_spaces = labyrinth.count(" ")
    num_food = labyrinth.count(".")
    num_squares = blank_spaces + num_food
    print labyrinth + "\n" + "\n"

    food_at = []
    pacman_at = ()
    adjacencies = {}

    square_symbols = (" ", ".")

    for row_idx, row in enumerate(labyrinth_list):
        for col_idx, square in enumerate(row):
            if square == "%":
                pass
            else:
                if square == ".":
                    food_at += ["sq" + str(row_idx) + "_" + str(col_idx)]
                if square == "P":
                    pacman_at = ["sq" + str(row_idx) + "_" + str(col_idx)]
                adjacencies["sq" + str(row_idx) + "_" + str(col_idx)] = []
                if (row_idx > 0) and (labyrinth_list[row_idx - 1][col_idx] in square_symbols):
                    adjacencies["sq" + str(row_idx) + "_" + str(col_idx)] += ["sq" + str(row_idx - 1) + "_" + str(col_idx)]
                if (row_idx < (len(labyrinth_list) - 1)) and (labyrinth_list[row_idx + 1][col_idx] in square_symbols):
                    adjacencies["sq" + str(row_idx) + "_" + str(col_idx)] += ["sq" + str(row_idx + 1) + "_" + str(col_idx)]
                if (col_idx > 0) and (labyrinth_list[row_idx][col_idx - 1] in square_symbols):
                    adjacencies["sq" + str(row_idx) + "_" + str(col_idx)] += ["sq" + str(row_idx) + "_" + str(col_idx - 1)]
                if (col_idx < (len(row) - 1)) and (labyrinth_list[row_idx][col_idx + 1] in square_symbols):
                    adjacencies["sq" + str(row_idx) + "_" + str(col_idx)] += ["sq" + str(row_idx) + "_" + str(col_idx + 1)]
            sys.stdout.write(square)
            sys.stdout.flush()

        sys.stdout.write('\n')
        sys.stdout.flush()

    return food_at, pacman_at, adjacencies

def create_pddl_file(food_at, pacman_at, adjacencies, out_file):
    path, filename = os.path.split(out_file)
    filename, file_extension = os.path.splitext(filename)
    squares_names = [key for key, value in adjacencies.items()]
    with open(out_file, "w+") as out_pddl_file:
        out_pddl_file.write("(define (problem " + filename + ")\n")
        out_pddl_file.write("\t(:domain pacman)\n")

        # Define objects
        out_pddl_file.write("\t(:objects\n")
        out_pddl_file.write("\t\t " + " ".join(squares_names) + " - position\n")
        out_pddl_file.write("\t)\n")

        out_pddl_file.write("\t(:init\n")
        # Pacman position
        out_pddl_file.write("\t\t(pacman-at-position " + pacman_at[0] + ")\n")

        # Position of food
        for square in squares_names:
            if square not in food_at:
                out_pddl_file.write("\t\t(empty-at-position " + square + ")\n")

        # Adjacencies
        for square, adjacent_squares in adjacencies.items():
            for adjacent_square in adjacent_squares:
                out_pddl_file.write("\t\t(adjacent " + square + " " + adjacent_square + ")\n")
        out_pddl_file.write("\t)\n") # close init

        # Define goal
        out_pddl_file.write("\t(:goal\n")
        out_pddl_file.write("\t\t(and\n")
        for square in food_at:
            out_pddl_file.write("\t\t(empty-at-position " + square + ")\n")
        out_pddl_file.write("\t\t)\n") # close and
        out_pddl_file.write("\t)\n") # close goal

        out_pddl_file.write(")")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--labyrinth_file", dest="labyrinth_file")
    parser.add_argument("--out_file", dest="out_file")
    args = parser.parse_args()
    food_at, pacman_at, adjacencies = parse_labyrinth(args.labyrinth_file)
    create_pddl_file(food_at, pacman_at, adjacencies, args.out_file)