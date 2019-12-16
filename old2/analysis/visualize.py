import sys
sys.path.append("..")
import result    #@UnresolvedImport
import tools

PREFIX = "../results/result"
SUFFIX = ".p"

idx = sys.argv[1]
#idx = 10

filename = PREFIX + str(idx) + SUFFIX

res = result.Result.load_from_file(filename)

tools.preview_input2(res.testing_sequence, slow_features=res.testing_SFA2_output_simple, retrieved_sequence=res.testing_SFA2_output_episodic, dimensions=4, rate=10)