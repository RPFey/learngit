class TreeNode:
    def __init__(self, key = 0):
        self.val = key
        self.left = None
        self.right = None
     
class ThreadedNode:
    def __init__(self, key = 0):
        self.val = key
        self.left = None
        self.right = None
        self.successor = 1

def stringToTreeNode(input):
    input = input.strip()
    input = input[1:-1]
    if not input:
        return None

    inputValues = [s.strip() for s in input.split(',')]
    root = TreeNode(int(inputValues[0]))
    nodeQueue = [root]
    front = 0
    index = 1
    while index < len(inputValues):
        node = nodeQueue[front]
        front = front + 1

        item = inputValues[index]
        index = index + 1
        if item != "null":
            leftNumber = int(item)
            node.left = TreeNode(leftNumber)
            nodeQueue.append(node.left)

        if index >= len(inputValues):
            break

        item = inputValues[index]
        index = index + 1
        if item != "null":
            rightNumber = int(item)
            node.right = TreeNode(rightNumber)
            nodeQueue.append(node.right)
    return root

def ConstructThreadedTree(input):
    input = input.strip()
    input = input[1:-1]
    if not input:
        return None

    inputValues = [s.strip() for s in input.split(',')]
    root = ThreadedNode(int(inputValues[0]))
    nodeQueue = [root]
    front = 0
    index = 1
    while index < len(inputValues):
        node = nodeQueue[front]
        front = front + 1

        item = inputValues[index]
        index = index + 1
        if item != "null":
            leftNumber = int(item)
            node.left = ThreadedNode(leftNumber)
            nodeQueue.append(node.left)

        if index >= len(inputValues):
            break

        item = inputValues[index]
        index = index + 1
        if item != "null":
            rightNumber = int(item)
            node.right = ThreadedNode(rightNumber)
            nodeQueue.append(node.right)
    
    nodeQueue.clear()
    p = root
    while p is not None:
        while p is not None:
            if p.right is not None:
                nodeQueue.append(p.right)
            nodeQueue.append(p)
            p = p.left
        p = nodeQueue.pop()
        while p.right is None and len(nodeQueue)!=0:
            p.right = nodeQueue[-1]
            p = nodeQueue.pop()
        p.successor = 0
        if nodeQueue:
            p = nodeQueue.pop()
        else :
            p = None
    return root

def integerListToString(nums, len_of_list=None):
    if not len_of_list:
        len_of_list = len(nums)
    return json.dumps(nums[:len_of_list])