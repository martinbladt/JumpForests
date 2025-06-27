# Contains classes for nodes and trees as well as functions to grow trees

library(R6)

# Classes
#---------------------------------------------------------------------------

NodeSurvival <- R6Class("NodeSurvival",
  public = list(
    feature = NULL,     # index of the feature being split upon
    threshold = NULL,   # threshold value for the split
    left = NULL,        # left daughter
    right = NULL,       # right daughter
    depth = NULL,       # depth of the node
    num = NULL,         # number of observations in the node
    value = NULL,       # predicted value of the node

    initialize = function(feature = NULL, threshold = NULL, depth = NULL) {
      self$feature <- feauture
      self$threshold <- threshold
      self$depth <- depth
    },

    print = function() {
      if (is.null(left)) {
        print("Node is a leaf")
      }
      else {
        print(paste("Feature:", feature))
        print(paste("Threshold:", threshold))
      }
      print(paste("Node depth:", depth))
      print(paste("#Observations:", num))
    }
  )
)

TreeSurvival <- R6Class("TreeSurvival",
  public <- list(
    root = NULL,
    num_leafs = NULL,

    initialize = function(root = NULL, num_leafs = NULL) {
      self$root <- root
    }
  )
)

