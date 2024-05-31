from pymol import cmd, stored, preset

def visualize(ligfile, recfile=None, use_surface=False):
    
    cmd.reinitialize("everything")

    cmd.load(ligfile, "lig")
    cmd.load(recfile, "rec")

    cmd.hide("all")

    preset.ball_and_stick(selection="lig", mode=1)
    
    if recfile:
        cmd.select("residues", "byres (rec within 3.5 of lig)")
        cmd.show("cartoon", "rec")
        cmd.show("sticks", "residues")

    cmd.color("grey30", "elem C")
    cmd.color("blue", "elem N")
    cmd.color("red", "elem O")
    cmd.color("yellow", "elem S")

    cmd.label("lig", "'%s' % (elem)")
    cmd.set("label_color", "black", "lig")
    cmd.set("transparency", 0.2, "residues")

    if use_surface:
        cmd.create("lig_surface", "lig")
        cmd.hide("everything", "lig_surface")
        cmd.show("surface", "lig_surface")
        cmd.set("solvent_radius", 0.05, "lig_surface")
        cmd.set("surface_quality", 2)
        cmd.set("surface_type", 4)
        cmd.set("transparency", 0.3, "lig_surface")

    cmd.set("antialias", 3)
    cmd.set("depth_cue", 0)
    cmd.set("spec_reflect", 0)
    cmd.set("ray_trace_mode", 1)
   
    cmd.set("bg_rgb", "white")
    cmd.set("valence", 1)
    cmd.set("valence_mode", 0)

    stored.bfs = []
    cmd.iterate("lig", "stored.bfs.append(b)")
    
    Bmax = max(max(stored.bfs), abs(min(stored.bfs)))

    if use_surface:
        cmd.spectrum("b", "orange_white_blue", "lig_surface", -Bmax, Bmax)
    else:
        cmd.spectrum("b", "orange_white_blue", "lig", -Bmax, Bmax)
        cmd.spectrum("b", "red_white_green", "residues", -Bmax, Bmax)

    cmd.center("lig")
    cmd.zoom("lig", 5)
    
cmd.extend("visualize", visualize)
