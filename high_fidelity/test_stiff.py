import os
import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

def run_test(mcp_k, pip_k, dip_k):
    xml = open(os.path.join(HERE, 'finger.xml')).read()
    
    # We will just forcefully replace the stiffnesses in the XML string for testing
    import re
    xml = re.sub(r'<joint name="mcp".*?stiffness="[\d.]+".*?/>', f'<joint name="mcp" stiffness="{mcp_k}" damping="0.08" range="-0.08727 1.57080"/>', xml)
    xml = re.sub(r'<joint name="pip".*?stiffness="[\d.]+".*?/>', f'<joint name="pip" stiffness="{pip_k}" damping="0.08" range="0.00000 1.91986"/>', xml)
    xml = re.sub(r'<joint name="dip".*?stiffness="[\d.]+".*?/>', f'<joint name="dip" stiffness="{dip_k}" damping="0.08" range="0.00000 1.57080"/>', xml)
    
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    
    mujoco.mj_forward(model, data)
    rest_len = data.ten_length[0]
    
    # Set delta L
    delta_L = 0.02 # 20mm
    model.tendon_lengthspring[0, :] = rest_len - delta_L
    
    # Step to equilibrium
    for _ in range(5000):
        mujoco.mj_step(model, data)
        
    mcp = np.degrees(data.qpos[0])
    pip = np.degrees(data.qpos[1])
    dip = np.degrees(data.qpos[2])
    print(f"K=({mcp_k}, {pip_k}, {dip_k}) -> Angles: MCP={mcp:.1f}, PIP={pip:.1f}, DIP={dip:.1f}")

run_test(1.0, 1.0, 1.0)
run_test(10.0, 1.0, 1.0)
run_test(1.0, 10.0, 1.0)
