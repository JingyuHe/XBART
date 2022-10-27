#' Get point-estimates of treatment effect (in sample)
#'
#' @param fit An XBCF fit object.
#'
#' @return An array of treatment effect point estimates.
#' @export
getTaus <- function(fit) {
    if(class(fit) != "XBCF")
        stop("Can only get taus for an XBCF object.")
    else
        tauhats <- rowMeans(fit$tauhats.adjusted)

    return(tauhats)
}

#' Get point-estimates of prognostic effect (in sample)
#'
#' @param fit An XBCF fit object.
#'
#' @return An array of prognostic effect point estimates.
#' @export
getMus <- function(fit) {
    if(class(fit) != "XBCF")
        stop("Can only get taus for an XBCF object.")
    else
        muhats <- rowMeans(fit$muhats.adjusted)

    return(muhats)
}