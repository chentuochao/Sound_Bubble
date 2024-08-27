import torch


"""
Flatten and unflatten context buffers
"""

DELIMITER = '::'

def flatten_state_buffers(state_buffers_dict, prefix=''):
    state_names = sorted(list(state_buffers_dict.keys()))
    
    res = []
    for k in state_names:
        if type(state_buffers_dict[k]) == dict:
            names, bufs = flatten_state_buffers(state_buffers_dict[k], prefix=f'{prefix}{k}{DELIMITER}')
            res.extend(list(zip(names, bufs)))
        else:
            assert type(state_buffers_dict[k] ) == torch.Tensor, f"Expected torch.Tensor, found {type(state_buffers_dict[k])}"
            res.append((f'{prefix}{k}', state_buffers_dict[k].clone()))
        
    buffer_names = [x[0] for x in res]
    buffers = [x[1] for x in res]
    
    return buffer_names, buffers

def unflatten_state_buffers(state_names, state_buffers):
    buffer_dict = {}

    # Initialize tree
    nodes = {'root':{'children': set()}}
    
    # Go over states
    for i, state_name in enumerate(state_names):
        # Get path
        node_list = state_name.split(DELIMITER)
        
        # Add link from root node to first node in path
        nodes['root']['children'].add(node_list[0])

        node_names = [DELIMITER.join(node_list[:j+1]) for j in range(len(node_list))]
        
        # Go over nodes in path
        for j in range(len(node_list)):
            
            # Initialize node if we haven't already
            if node_names[j] not in nodes:
                nodes[node_names[j]] = dict(children=set())

            # Add link from node to one after it in the path
            if j < len(node_list) - 1:
                nodes[node_names[j]]['children'].add(node_names[j+1])
    
        # Save state for last node
        nodes[node_names[-1]]['buf'] = state_buffers[i]

    def dfs(current_node, buffer_dict = {}):
        children = nodes[current_node]['children']
        name = current_node.split(DELIMITER)[-1]
        
        if len(children) == 0:
            buffer_dict[name] = nodes[current_node]['buf'].clone()
        else:
            buffer_dict[name] = {}
            for c in children:
                dfs(c, buffer_dict[name])

    current_node = 'root'
    dfs(current_node, buffer_dict)

    return buffer_dict['root']