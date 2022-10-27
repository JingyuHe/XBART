load.XBCF <- function(fileName) {
    json_str = readChar(fileName, file.info(fileName)$size)
    obj = .Call(`_XBART_json_to_r`, json_str)  # model$tree_pnt
    class(obj) = "XBCF"
    return(obj)
}