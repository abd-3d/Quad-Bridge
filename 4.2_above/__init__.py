import bpy
import bmesh
import gpu
from gpu_extras.batch import batch_for_shader
from mathutils import Vector


class QB_Flash_Effect:
    def __init__(self, context, vertices_tris, vertices_lines):
        # Updated for Blender 4.0+: Use 'UNIFORM_COLOR' instead of '3D_UNIFORM_COLOR'
        self.shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        self.vertices_tris = vertices_tris
        self.vertices_lines = vertices_lines
        
        self.batch_tris = batch_for_shader(
            self.shader, 'TRIS', {"pos": self.vertices_tris}
        )
        
        self.batch_lines = batch_for_shader(
            self.shader, 'LINES', {"pos": self.vertices_lines}
        )
        
        self._handle = bpy.types.SpaceView3D.draw_handler_add(
            self.draw_callback, (), 'WINDOW', 'POST_VIEW'
        )
        
        bpy.app.timers.register(self.kill_effect, first_interval=1.0)
        self.redraw_view3d(context)

    def draw_callback(self):
        package = __package__ if __package__ else "quad_bridge_standalone"
        base_color = (0.0, 0.8, 1.0)
        
        try:
            prefs = bpy.context.preferences.addons[package].preferences
            c = prefs.highlight_color
            base_color = (c[0], c[1], c[2])
        except (KeyError, AttributeError):
            pass

        gpu.state.blend_set('ALPHA')
        self.shader.bind()
        
        self.shader.uniform_float("color", (*base_color, 0.3))
        self.batch_tris.draw(self.shader)
        
        self.shader.uniform_float("color", (*base_color, 0.9))
        gpu.state.line_width_set(2.0)
        self.batch_lines.draw(self.shader)
        
        gpu.state.line_width_set(1.0)
        gpu.state.blend_set('NONE')

    def kill_effect(self):
        if self._handle:
            bpy.types.SpaceView3D.draw_handler_remove(self._handle, 'WINDOW')
            self._handle = None
            for window in bpy.context.window_manager.windows:
                for area in window.screen.areas:
                    if area.type == 'VIEW_3D':
                        area.tag_redraw()
        return None

    def redraw_view3d(self, context):
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()


class QuadBridgePreferences(bpy.types.AddonPreferences):
    bl_idname = __package__ if __package__ else "quad_bridge_standalone"

    highlight_color: bpy.props.FloatVectorProperty(
        name="Highlight Color",
        subtype='COLOR',
        default=(0.0, 0.8, 1.0, 1.0),
        size=4,
        min=0.0, max=1.0,
        description="Color of the visual feedback flash"
    )

    def draw(self, context):
        layout = self.layout
        box = layout.box()
        box.label(text="Visual Feedback Settings:")
        box.prop(self, "highlight_color")


def analyze_selection_type(bm, selected_edges):
    vert_links = {v: [] for e in selected_edges for v in e.verts}
    for e in selected_edges:
        for v in e.verts:
            vert_links[v].append(e)

    visited = set()
    islands = []

    for v in vert_links:
        if v not in visited:
            stack, component = [v], []
            while stack:
                curr = stack.pop()
                if curr in visited:
                    continue
                visited.add(curr)
                component.append(curr)
                for e in vert_links[curr]:
                    other = e.other_vert(curr)
                    if other not in visited:
                        stack.append(other)

            ends = [v for v in component if len(vert_links[v]) == 1]
            if not ends:
                continue
            ordered = [ends[0]]
            while len(ordered) < len(component):
                prev = ordered[-1]
                for e in vert_links[prev]:
                    nxt = e.other_vert(prev)
                    if nxt in component and nxt not in ordered:
                        ordered.append(nxt)
                        break
            islands.append(ordered)

    if len(islands) != 2:
        return "GENERAL", None, None

    c1, c2 = islands
    top, bot = (c1, c2) if len(c1) < len(c2) else (c2, c1)

    if (top[-1].co - top[0].co).dot(bot[-1].co - bot[0].co) < 0:
        bot.reverse()

    n_top, n_bot = len(top) - 1, len(bot) - 1

    topo_type = "GENERAL"
    if n_top == 1 and n_bot == 2:
        topo_type = "1_TO_2"
    elif abs(n_top - n_bot) == 2 and n_top > 1:
        topo_type = "GAP"

    return topo_type, top, bot


def get_chain_data(chain):
    lengths = []
    total = 0.0
    for i in range(len(chain) - 1):
        l = (chain[i + 1].co - chain[i].co).length
        lengths.append(l)
        total += l
    return lengths, total


def sample_chain_at_u(chain, u, lengths, total_length):
    if total_length == 0:
        return chain[0].co
    target_dist = u * total_length
    current_dist = 0.0
    for i, length in enumerate(lengths):
        if current_dist + length >= target_dist - 0.0001:
            factor = (target_dist - current_dist) / length if length > 0 else 0
            return chain[i].co.lerp(chain[i + 1].co, factor)
        current_dist += length
    return chain[-1].co


def get_u_at_index(index, lengths, total_length):
    if total_length == 0:
        return 0.0
    dist = sum(lengths[:index])
    return dist / total_length


def bridge_one_to_two_logic(bm, top, bot, method):
    t0, t1 = top[0], top[1]
    b0, b1, b2 = bot[0], bot[1], bot[2]

    if method == 0:
        m1 = bm.verts.new(t0.co.lerp(b1.co, 0.55))
        m2 = bm.verts.new(t1.co.lerp(b1.co, 0.55))
        bm.verts.ensure_lookup_table()
        bm.faces.new((t0, m1, b1, b0))
        bm.faces.new((t1, b2, b1, m2))
        bm.faces.new((t0, t1, m2, m1))
        bm.faces.new((m1, m2, b1))
    elif method == 1:
        p_r = (t1.co + b2.co) * 0.5
        p_c = (t0.co + b1.co + p_r) / 3.0
        vr, vc = bm.verts.new(p_r), bm.verts.new(p_c)
        bm.verts.ensure_lookup_table()
        bm.faces.new((t0, b0, b1, vc))
        bm.faces.new((vc, b1, b2, vr))
        bm.faces.new((t0, vc, vr, t1))
    elif method == 2:
        p_l = (t0.co + b0.co) * 0.5
        p_c = (t1.co + b1.co + p_l) / 3.0
        vl, vc = bm.verts.new(p_l), bm.verts.new(p_c)
        bm.verts.ensure_lookup_table()
        bm.faces.new((t1, vc, b1, b2))
        bm.faces.new((vl, b0, b1, vc))
        bm.faces.new((t0, vl, vc, t1))


def bridge_odd_gap_outer(bm, top, bot, n_top, n_bot):
    mid_verts = []
    tl, tt = get_chain_data(top)
    bl, bt = get_chain_data(bot)
    for i in range(n_top + 1):
        u_bot = get_u_at_index(i + 1, bl, bt)
        v_top = sample_chain_at_u(top, u_bot, tl, tt)
        mid_verts.append(bm.verts.new((v_top + bot[i + 1].co) / 2))
    bm.verts.ensure_lookup_table()
    bm.faces.new((top[0], bot[0], bot[1], mid_verts[0]))
    bm.faces.new((mid_verts[-1], bot[-2], bot[-1], top[-1]))
    for i in range(n_top):
        bm.faces.new((top[i], mid_verts[i], mid_verts[i + 1], top[i + 1]))
        bm.faces.new((mid_verts[i], bot[i + 1], bot[i + 2], mid_verts[i + 1]))


def bridge_odd_gap_inner(bm, top, bot, n_top, n_bot):
    top_center_left = n_top // 2
    top_center_right = top_center_left + 1
    bot_center_left = n_bot // 2
    bot_center_right = bot_center_left + 1

    v15 = bm.verts.new((top[top_center_left].co + bot[bot_center_left].co) / 2)
    v16 = bm.verts.new((top[top_center_right].co + bot[bot_center_right].co) / 2)

    for i in range(top_center_left):
        if i < bot_center_left:
            bm.faces.new((top[i], top[i + 1], bot[i + 1], bot[i]))
        else:
            if i == top_center_left - 1:
                bm.faces.new((top[i], top[i + 1], v15, bot[i]))
            else:
                bm.faces.new((top[i], top[i + 1], bot[i + 1], bot[i]))

    for i in range(n_top - 1, top_center_right - 1, -1):
        right_bot_idx = i - (n_top - n_bot)
        if right_bot_idx < n_bot - 1 and right_bot_idx >= bot_center_right:
            bm.faces.new((top[i], bot[right_bot_idx], bot[right_bot_idx + 1], top[i + 1]))
        else:
            if i == top_center_right:
                bm.faces.new((top[i], v16, bot[bot_center_right], top[i + 1]))
            elif right_bot_idx >= 0 and right_bot_idx < n_bot:
                bm.faces.new((top[i], bot[right_bot_idx], bot[right_bot_idx + 1], top[i + 1]))

    bm.faces.new((top[top_center_left], top[top_center_right], v16, v15))
    bm.faces.new((v15, v16, bot[bot_center_right], bot[bot_center_left]))


def bridge_even_gap_two_outer(bm, top, bot, n_top, n_bot):
    mid_verts = []
    tl, tt = get_chain_data(top)
    bl, bt = get_chain_data(bot)
    for i in range(len(top)):
        u = get_u_at_index(i + 1, bl, bt)
        vt = sample_chain_at_u(top, u, tl, tt)
        mid_verts.append(bm.verts.new((vt + bot[i + 1].co) / 2))
    bm.verts.ensure_lookup_table()
    for i in range(n_top):
        bm.faces.new((top[i], top[i + 1], mid_verts[i + 1], mid_verts[i]))
    for i in range(n_top):
        bm.faces.new((mid_verts[i], mid_verts[i + 1], bot[i + 2], bot[i + 1]))
    bm.faces.new((top[0], mid_verts[0], bot[1], bot[0]))
    bm.faces.new((top[-1], bot[-1], bot[-2], mid_verts[-1]))


def bridge_even_gap_two_inner(bm, top, bot, n_top, n_bot):
    side = (n_top - 2) // 2
    for i in range(side):
        bm.faces.new((top[i], top[i + 1], bot[i + 1], bot[i]))
    ts, bs = n_top - side, n_bot - side
    for i in range(side):
        bm.faces.new((top[ts + i], top[ts + i + 1], bot[bs + i + 1], bot[bs + i]))
    bridge_even_two_to_n(bm, top[side : ts + 1], bot[side : bs + 1], 4)


def bridge_even_gap_diamond(bm, top, bot, n_top, n_bot):
    mid = n_top // 2
    for i in range(mid):
        try:
            bm.faces.new((top[i], top[i+1], bot[i+1], bot[i]))
        except ValueError:
            pass
    try:
        bm.faces.new((top[mid], bot[mid+2], bot[mid+1], bot[mid]))
    except ValueError:
        pass
    for i in range(mid, n_top):
        try:
            bm.faces.new((top[i], top[i+1], bot[i+3], bot[i+2]))
        except ValueError:
            pass


def bridge_odd_one_to_n(bm, top, bot, n_bot):
    if n_bot == 1:
        bm.faces.new((top[0], bot[0], bot[1], top[1]))
        return
    rows = []
    nr = (n_bot - 1) // 2
    steps = nr + 1
    for i in range(nr):
        t = (i + 1) / steps
        bi_l, bi_r = i + 1, n_bot - 1 - i
        u_l, u_r = bi_l / n_bot, bi_r / n_bot
        ta_l = top[0].co.lerp(top[-1].co, u_l)
        ta_r = top[0].co.lerp(top[-1].co, u_r)
        rows.append((bm.verts.new(ta_l.lerp(bot[bi_l].co, t)), bm.verts.new(ta_r.lerp(bot[bi_r].co, t))))
    bm.verts.ensure_lookup_table()
    bm.faces.new((top[0], rows[0][0], rows[0][1], top[1]))
    for i in range(len(rows) - 1):
        bm.faces.new((rows[i][0], rows[i + 1][0], rows[i + 1][1], rows[i][1]))
    for i in range(len(rows)):
        if i == 0:
            bm.faces.new((top[0], bot[0], bot[1], rows[0][0]))
            bm.faces.new((rows[0][1], bot[n_bot - 1], bot[n_bot], top[1]))
        else:
            bm.faces.new((rows[i - 1][0], bot[i], bot[i + 1], rows[i][0]))
            bm.faces.new((rows[i][1], bot[n_bot - i - 1], bot[n_bot - i], rows[i - 1][1]))
    bm.faces.new((rows[-1][0], bot[nr], bot[nr + 1], rows[-1][1]))


def bridge_even_one_to_n(bm, top, bot, n_bot):
    nr = n_bot // 2
    rows = []
    for i in range(nr - 1):
        t = (i + 1) / nr
        bi_l, bi_r = i + 1, n_bot - 1 - i
        u_l, u_r = bi_l / n_bot, bi_r / n_bot
        ta_l = top[0].co.lerp(top[-1].co, u_l)
        ta_r = top[0].co.lerp(top[-1].co, u_r)
        rows.append((bm.verts.new(ta_l.lerp(bot[bi_l].co, t)), bm.verts.new(ta_r.lerp(bot[bi_r].co, t))))
    bm.verts.ensure_lookup_table()
    bm.faces.new((top[0], rows[0][0], rows[0][1], top[1]))
    for i in range(len(rows) - 1):
        bm.faces.new((rows[i][0], rows[i + 1][0], rows[i + 1][1], rows[i][1]))
    for i in range(len(rows)):
        if i == 0:
            bm.faces.new((top[0], bot[0], bot[1], rows[0][0]))
            bm.faces.new((rows[0][1], bot[n_bot - 1], bot[n_bot], top[1]))
        else:
            bm.faces.new((rows[i - 1][0], bot[i], bot[i + 1], rows[i][0]))
            bm.faces.new((rows[i][1], bot[n_bot - i - 1], bot[n_bot - i], rows[i - 1][1]))
    mid = n_bot // 2
    bm.faces.new((rows[-1][0], bot[mid - 1], bot[mid]))
    bm.faces.new((rows[-1][1], bot[mid], bot[mid + 1]))
    bm.faces.new((rows[-1][0], bot[mid], rows[-1][1]))


def bridge_even_two_to_n(bm, top, bot, n_bot):
    if n_bot == 2:
        bm.faces.new((top[0], top[1], bot[1], bot[0]))
        bm.faces.new((top[1], top[2], bot[2], bot[1]))
        return
    nr = (n_bot - 2) // 2
    rows = []
    mid = n_bot // 2
    tl, tt = get_chain_data(top)
    bl, bt = get_chain_data(bot)
    steps = nr + 1
    for i in range(nr):
        t = (i + 1) / steps
        bi_l = i + 1
        u_l = get_u_at_index(bi_l, bl, bt)
        ta_l = sample_chain_at_u(top, u_l, tl, tt)
        v0 = bm.verts.new(ta_l.lerp(bot[bi_l].co, t))
        v1 = bm.verts.new(top[1].co.lerp(bot[mid].co, t))
        bi_r = n_bot - 1 - i
        u_r = get_u_at_index(bi_r, bl, bt)
        ta_r = sample_chain_at_u(top, u_r, tl, tt)
        v2 = bm.verts.new(ta_r.lerp(bot[bi_r].co, t))
        rows.append((v0, v1, v2))
    bm.verts.ensure_lookup_table()
    if rows:
        bm.faces.new((top[0], rows[0][0], rows[0][1], top[1]))
        bm.faces.new((top[1], rows[0][1], rows[0][2], top[2]))
        for i in range(len(rows) - 1):
            curr, nxt = rows[i], rows[i + 1]
            bm.faces.new((curr[0], nxt[0], nxt[1], curr[1]))
            bm.faces.new((curr[1], nxt[1], nxt[2], curr[2]))
        last = rows[-1]
        bm.faces.new((last[0], bot[mid - 1], bot[mid], last[1]))
        bm.faces.new((last[1], bot[mid], bot[mid + 1], last[2]))
    l_chain = [top[0]] + [r[0] for r in rows]
    for i in range(nr):
        bm.faces.new((l_chain[i], bot[i], bot[i + 1], l_chain[i + 1]))
    r_chain = [top[2]] + [r[2] for r in rows]
    for i in range(nr):
        bm.faces.new((r_chain[i], r_chain[i + 1], bot[n_bot - 1 - i], bot[n_bot - i]))


def bridge_odd_two_to_n(bm, top, bot, n_bot):
    def get_top_u(u):
        return top[0].co.lerp(top[1].co, u / 0.5) if u <= 0.5 else top[1].co.lerp(top[2].co, (u - 0.5) / 0.5)

    u_l, u_r = 1.0 / n_bot, (n_bot - 1) / n_bot
    mid_L = bm.verts.new(top[0].co.lerp(bot[0].co, 0.5))
    mid_R = bm.verts.new(top[2].co.lerp(bot[-1].co, 0.5))
    tri_L = bm.verts.new(get_top_u(u_l).lerp(bot[1].co, 0.5))
    tri_R = bm.verts.new(get_top_u(u_r).lerp(bot[-2].co, 0.5))
    bm.verts.ensure_lookup_table()
    bm.faces.new((top[0], top[1], tri_L, mid_L))
    bm.faces.new((mid_L, tri_L, bot[1], bot[0]))
    bm.faces.new((top[1], top[2], mid_R, tri_R))
    bm.faces.new((tri_R, mid_R, bot[-1], bot[-2]))
    bm.faces.new((top[1], tri_R, tri_L))
    if n_bot - 2 == 1:
        bm.faces.new((tri_L, tri_R, bot[2], bot[1]))
    else:
        bridge_odd_one_to_n(bm, [tri_L, tri_R], bot[1:-1], n_bot - 2)


def bridge_general_n_m(bm, top, bot, n_top, n_bot, flow_1to2_method):
    if n_top % 2 == 0 and n_bot % 2 == 1:
        side_len = (n_top - 2) // 2
        mid_l_anchor, mid_r_anchor = None, None
        prev_mid = None
        for i in range(side_len):
            if prev_mid:
                m_curr = prev_mid
            else:
                m_curr = bm.verts.new((top[i].co + bot[i].co) / 2)
            m_next = bm.verts.new((top[i + 1].co + bot[i + 1].co) / 2)
            bm.faces.new((top[i], top[i + 1], m_next, m_curr))
            bm.faces.new((m_curr, m_next, bot[i + 1], bot[i]))
            prev_mid = m_next
            if i == side_len - 1:
                mid_l_anchor = m_next
        ts, bs = n_top - side_len, n_bot - side_len
        prev_mid = None
        for i in range(side_len):
            t_idx, b_idx = ts + i, bs + i
            if prev_mid:
                m_curr = prev_mid
            else:
                m_curr = bm.verts.new((top[t_idx].co + bot[b_idx].co) / 2)
                if i == 0:
                    mid_r_anchor = m_curr
            m_next = bm.verts.new((top[t_idx + 1].co + bot[b_idx + 1].co) / 2)
            bm.faces.new((top[t_idx], top[t_idx + 1], m_next, m_curr))
            bm.faces.new((m_curr, m_next, bot[b_idx + 1], bot[b_idx]))
            prev_mid = m_next
        c_top, c_bot = top[side_len : ts + 1], bot[side_len : bs + 1]
        
        def get_ct_u(u):
            return c_top[0].co.lerp(c_top[1].co, u / 0.5) if u <= 0.5 else c_top[1].co.lerp(c_top[2].co, (u - 0.5) / 0.5)
        
        cn_bot = len(c_bot) - 1
        u_l, u_r = 1.0 / cn_bot, (cn_bot - 1) / cn_bot
        if not mid_l_anchor:
            mid_l_anchor = bm.verts.new((c_top[0].co + c_bot[0].co) / 2)
        if not mid_r_anchor:
            mid_r_anchor = bm.verts.new((c_top[2].co + c_bot[-1].co) / 2)
        tri_l = bm.verts.new(get_ct_u(u_l).lerp(c_bot[1].co, 0.5))
        tri_r = bm.verts.new(get_ct_u(u_r).lerp(c_bot[-2].co, 0.5))
        bm.verts.ensure_lookup_table()
        bm.faces.new((c_top[0], c_top[1], tri_l, mid_l_anchor))
        bm.faces.new((mid_l_anchor, tri_l, c_bot[1], c_bot[0]))
        bm.faces.new((c_top[1], c_top[2], mid_r_anchor, tri_r))
        bm.faces.new((tri_r, mid_r_anchor, c_bot[-1], c_bot[-2]))
        bm.faces.new((c_top[1], tri_r, tri_l))
        rem_hole_edges = cn_bot - 2
        if rem_hole_edges == 1:
            bm.faces.new((tri_l, tri_r, c_bot[2], c_bot[1]))
        else:
            bridge_odd_one_to_n(bm, [tri_l, tri_r], c_bot[1:-1], rem_hole_edges)
        return

    is_top_odd = n_top % 2 == 1
    side_len = (n_top - 1) // 2 if is_top_odd else (n_top - 2) // 2
    for i in range(side_len):
        bm.faces.new((top[i], top[i + 1], bot[i + 1], bot[i]))
    ts, bs = n_top - side_len, n_bot - side_len
    for i in range(side_len):
        bm.faces.new((top[ts + i], top[ts + i + 1], bot[bs + i + 1], bot[bs + i]))
    c_top, c_bot = top[side_len : ts + 1], bot[side_len : bs + 1]
    cn_bot = len(c_bot) - 1
    center_method = flow_1to2_method
    if len(c_top) - 1 == 1 and cn_bot == 2:
        center_method = 0
    if is_top_odd:
        if cn_bot == 2:
            bridge_one_to_two_logic(bm, c_top, c_bot, center_method)
        elif cn_bot % 2 == 0:
            bridge_even_one_to_n(bm, c_top, c_bot, cn_bot)
        else:
            bridge_odd_one_to_n(bm, c_top, c_bot, cn_bot)
    else:
        bridge_even_two_to_n(bm, c_top, c_bot, cn_bot)


class MESH_OT_QuadBridge(bpy.types.Operator):
    bl_idname = "mesh.quad_bridge"
    bl_label = "Quad Bridge"
    bl_options = {"REGISTER", "UNDO"}

    loop_method: bpy.props.EnumProperty(
        name="Gap Method",
        items=[("0", "Outer Loop", ""), ("1", "Inner Loop", ""), ("2", "Diamond", "")],
        default="0",
    )
    
    loop_method_simple: bpy.props.EnumProperty(
        name="Gap Method",
        items=[("0", "Outer Loop", ""), ("1", "Inner Loop", "")],
        default="0",
    )
    
    flow_method: bpy.props.EnumProperty(
        name="Flow",
        items=[("0", "Diamond", ""), ("1", "Left Flow", ""), ("2", "Right Flow", "")],
        default="0",
    )
    
    topo_type: bpy.props.StringProperty(default="GENERAL")
    is_odd_gap: bpy.props.BoolProperty(default=False)

    def invoke(self, context, event):
        if not context.edit_object:
            return {"CANCELLED"}
        bm = bmesh.from_edit_mesh(context.edit_object.data)
        selected_edges = [e for e in bm.edges if e.select]
        if not selected_edges:
            return self.execute(context)
        
        t_type, top, bot = analyze_selection_type(bm, selected_edges)
        self.is_odd_gap = False
        should_popup = False
        
        if t_type == "1_TO_2":
            should_popup = True
        elif t_type == "GAP":
            should_popup = True
            n_top = len(top) - 1
            if n_top % 2 == 1:
                self.is_odd_gap = True

        if t_type:
            self.topo_type = t_type
        if should_popup:
            return context.window_manager.invoke_props_dialog(self)
        else:
            return self.execute(context)

    def draw(self, context):
        layout = self.layout
        if self.topo_type == "GAP":
            if self.is_odd_gap:
                layout.prop(self, "loop_method_simple", expand=True)
            else:
                layout.prop(self, "loop_method", expand=True)
        elif self.topo_type == "1_TO_2":
            layout.prop(self, "flow_method", expand=True)

    def execute(self, context):
        obj = bpy.context.edit_object
        if not obj:
            return {"CANCELLED"}
        me = obj.data
        bm = bmesh.from_edit_mesh(me)
        
        bm.faces.ensure_lookup_table()
        old_face_indices = {f.index for f in bm.faces}

        selected_edges = [e for e in bm.edges if e.select]
        if not selected_edges:
            return {"CANCELLED"}
        
        t_type, top, bot = analyze_selection_type(bm, selected_edges)
        if t_type:
            self.topo_type = t_type
        if not top:
            return {"CANCELLED"}
        
        n_top, n_bot = len(top) - 1, len(bot) - 1
        flow_m = int(self.flow_method)
        loop_m = int(self.loop_method_simple) if (n_top % 2 == 1) else int(self.loop_method)
        
        self.is_odd_gap = (n_top % 2 == 1)

        if t_type == "1_TO_2":
            bridge_one_to_two_logic(bm, top, bot, flow_m)
        elif t_type == "GAP":
            if n_top % 2 == 1:
                if loop_m == 0:
                    bridge_odd_gap_outer(bm, top, bot, n_top, n_bot)
                else:
                    bridge_odd_gap_inner(bm, bot, top, n_bot, n_top)
            else:
                if loop_m == 0:
                    bridge_even_gap_two_outer(bm, top, bot, n_top, n_bot)
                elif loop_m == 1:
                    bridge_even_gap_two_inner(bm, top, bot, n_top, n_bot)
                elif loop_m == 2:
                    bridge_even_gap_diamond(bm, top, bot, n_top, n_bot)
        elif n_top == 1 and n_bot == 3:
            v1, v2 = top
            v3, v4, v5, v6 = bot
            w = (v5.co - v4.co) * 0.5
            mc = ((v4.co + v5.co) / 2 + (v1.co + v2.co) / 2) / 2
            v7, v8 = bm.verts.new(mc - w), bm.verts.new(mc + w)
            bm.faces.new((v1, v3, v4, v7))
            bm.faces.new((v7, v4, v5, v8))
            bm.faces.new((v8, v5, v6, v2))
            bm.faces.new((v1, v7, v8, v2))
        elif n_top == 1 and n_bot % 2 == 1:
            bridge_odd_one_to_n(bm, top, bot, n_bot)
        elif n_top == 1 and n_bot % 2 == 0:
            bridge_even_one_to_n(bm, top, bot, n_bot)
        elif n_top == 2 and n_bot % 2 == 0:
            bridge_even_two_to_n(bm, top, bot, n_bot)
        elif n_top == 2 and n_bot % 2 == 1:
            bridge_odd_two_to_n(bm, top, bot, n_bot)
        else:
            bridge_general_n_m(bm, top, bot, n_top, n_bot, flow_m)

        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        
        bm.faces.ensure_lookup_table()
        new_faces = [f for f in bm.faces if f.index not in old_face_indices]
        
        if new_faces:
            mw = obj.matrix_world
            tris_coords = []
            lines_coords = []
            
            for f in new_faces:
                f_verts = [mw @ v.co for v in f.verts]
                if len(f_verts) >= 3:
                    v0 = f_verts[0]
                    for i in range(1, len(f_verts) - 1):
                        tris_coords.extend([v0, f_verts[i], f_verts[i+1]])
                
                for i in range(len(f_verts)):
                    lines_coords.extend([f_verts[i], f_verts[(i+1) % len(f_verts)]])
            
            if tris_coords:
                QB_Flash_Effect(context, tris_coords, lines_coords)

        bmesh.update_edit_mesh(me)
        return {"FINISHED"}


classes = (
    QuadBridgePreferences,
    MESH_OT_QuadBridge,
)


def menu_func(self, context):
    self.layout.operator(MESH_OT_QuadBridge.bl_idname, icon="MOD_WIREFRAME")


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.VIEW3D_MT_edit_mesh_context_menu.prepend(menu_func)


def unregister():
    bpy.types.VIEW3D_MT_edit_mesh_context_menu.remove(menu_func)
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
