check_scalar <- function(x, obj_name) {
    if (!(is.atomic(x) && length(x) == 1L)) {
        error_message = paste(obj_name, " should be a double object.\n")
        stop(error_message)
    }
}

check_positive_integer <- function(x, obj_name) {
    if (x <= 0 || length(x) != 1L) {
        error_message = paste(obj_name, " should be a positive integer.\n")
        
    }
}

check_non_negative_integer <- function(x, obj_name) {
    if (x < 0 || length(x) != 1L) {
        error_message = paste(obj_name, " should be a non-negative integer.\n")
        stop(error_message)
    }
}