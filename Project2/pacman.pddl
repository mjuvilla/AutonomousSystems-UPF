(define (domain pacman)

	(:requirements 
		:typing
	)

	(:types
		position
	)

	(:predicates
		(pacman-at-position ?x - position) (food-at-position ?x - position) (adjacent ?x ?y - position) 	
	) 

	(:action move
		:parameters (?current-position ?next-position - position)
		:precondition ( and (pacman-at-position ?current-position)
						(adjacent ?current-position ?next-position)
						)
		:effect (and (pacman-at-position ?next-position)
				(not (pacman-at-position ?current-position))
				(not (food-at-position ?next-position))
				)
	)
)

