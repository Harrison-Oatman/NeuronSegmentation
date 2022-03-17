# def get_process_cluster(process_image, rna):
#     RNAx = rna["global_x"]
#     RNAy = rna["global_y"]
#     RNAid = rna['barcode_id']
#     RNAcol = [barcodeColor[id] for id in RNAid]
#     processIndex = get_rna_process(rna, process_image)
#
#     clusters = np.unique(RNAcol)
#     processCluster = {}
#
#     # print(processIndex)
#     for i, process_test in enumerate(processIndex):
#         # print(process_test)
#         if process_test not in processCluster:
#             processCluster[process_test] = {}
#             for cluster in clusters:
#                 processCluster[process_test][cluster] = 0
#         processCluster[process_test][RNAcol[i]] += 1
#
#     return processCluster


# def getScaledClusterVal(process_image, clusters, color):
#     n = np.max(list(clusters.keys()))
#     relval = np.zeros(n+10)
#     for process_test in range(n):
#         if process_test in clusters:
#             tot = np.sum(list(clusters[process_test].values()))
#             # print(tot)
#             col = np.sum(clusters[process_test][color])
#             relval[process_test] = col/tot
#
#     relval = relval + 0.05
#
#     relval[0] = 0
#
#     def colorofprocess(process_test):
#         return relval[process_test]
#
#     print(np.max(process_image))
#     print(n)
#
#     v_get_val = np.vectorize(colorofprocess)
#
#     return v_get_val(process_image)


# def plotClusterImage(processIamge, clusters, color, cmap, bbs):
#     colored = getScaledClusterVal(process_image, clusters, color)
#     embedimg(colored, bbs, cmap=cmap,name="process_cluster_proportion_" + color)


# def getScaledClusters(clusters):
#     scaledClusters = {}
#     for i in clusters:
#         scaledClusters[i] = {}
#         tot = 5
#         for c in clusters[i]:
#             tot += clusters[i][c]
#         for c in clusters[i]:
#             scaledClusters[i][c] = clusters[i][c]/tot
#     return scaledClusters


# def getClusterDis(clusters, neighbors):
#     sc = getScaledClusters(clusters)
#
#     def distance(i, j):
#         if i in sc and j in sc:
#             ic = sc[i]
#             jc = sc[j]
#             diff = 0
#             for c in ic:
#                 if c is "black":
#                     continue
#                 diff += abs(ic[c] - jc[c])
#             return diff
#         else:
#             return 0.5
#
#     distances = {}
#
#     for i in neighbors.keys():
#         distances[i] = {}
#         for j in neighbors[i]:
#             dist = distance(i, j)
#             distances[i][j] = dist
#
#     return distances