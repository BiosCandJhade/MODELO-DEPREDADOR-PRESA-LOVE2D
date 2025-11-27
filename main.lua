local cfg = {
    prey_count = 100,
    pred_count = 15,
    episode_time = 300.0,
    vel_base_prey = 120,
    vel_base_pred = 160,
    stamina_max_prey = 1.0,
    drain_prey = 0.12,
    recover_prey = 0.06,
    stamina_max_pred = 1.0,
    drain_pred = 0.18,
    recover_pred = 0.04,
    vision_prey = 1000,
    vision_pred = 150,
    pmut = 0.05,
    pcross = 0.5,
    elitism = 0.05,
    H = 8,
    sprint_threshold = 0.7,
    comm_range = 120,
    coord_window = 1.2,
    input_size = 9,
    output_size = 3,
    prey_radius = 6,
    pred_size = 10,
    capture_radius = 12,
    world_w = 1200,
    world_h = 800,
    max_agents = 1000,
    tournament_k = 3,
    generation_size_limit = 500,
    hist_bins = 20,
    prey_survival_weight = 1.0,
    prey_esquivo_weight = 1.0,
    pred_capture_weight = 1.0,
    pred_rodeo_weight = 0.8
}

local util = {}
function util.rand(a,b) if b then return a + math.random()*(b-a) end return math.random() end
function util.clamp(x,a,b) if x<a then return a elseif x>b then return b else return x end end
function util.dist(x1,y1,x2,y2) local dx=x2-x1; local dy=y2-y1; return math.sqrt(dx*dx+dy*dy),dx,dy end
function util.normAngle(a) while a<=-math.pi do a=a+2*math.pi end while a>math.pi do a=a-2*math.pi end return a end
function util.copy(t) if type(t)~="table" then return t end local n={} for k,v in pairs(t) do n[k]=util.copy(v) end return n end
function util.randn(mu,sigma) local u1=math.random(); local u2=math.random(); local z=math.sqrt(-2*math.log(u1))*math.cos(2*math.pi*u2); return mu + z * sigma end

local NN = {}
function NN.newGenome(input,hid,output)
    local g = {I=input,H=hid,O=output,wh={},bh={},wo={},bo={}}
    for i=1,hid do
        g.bh[i] = util.rand(-1,1)
        g.wh[i] = {}
        for j=1,input do g.wh[i][j] = util.rand(-1,1) end
    end
    for i=1,output do
        g.bo[i] = util.rand(-1,1)
        g.wo[i] = {}
        for j=1,hid do g.wo[i][j] = util.rand(-1,1) end
    end
    return g
end

function NN.forward(g,inputs)
    local hout = {}
    for i=1,g.H do
        local s = g.bh[i] or 0
        for j=1,g.I do s = s + (g.wh[i][j] or 0) * (inputs[j] or 0) end
        hout[i] = math.tanh(s)
    end
    local out = {}
    for i=1,g.O do
        local s = g.bo[i] or 0
        for j=1,g.H do s = s + (g.wo[i][j] or 0) * hout[j] end
        out[i] = math.tanh(s)
    end
    return out
end

function NN.clone(g) return util.copy(g) end

function NN.crossover(a,b,pcross)
    local c = NN.clone(a)
    for i=1,a.H do
        for j=1,a.I do
            if math.random() < pcross then
                local w1 = a.wh[i][j] or 0; local w2 = b.wh[i][j] or 0
                local alpha = util.rand(0,1)
                c.wh[i][j] = alpha*w1 + (1-alpha)*w2
            end
        end
        if math.random() < pcross then
            local alpha = util.rand(0,1)
            c.bh[i] = alpha*(a.bh[i] or 0) + (1-alpha)*(b.bh[i] or 0)
        end
    end
    for i=1,a.O do
        for j=1,a.H do
            if math.random() < pcross then
                local w1 = a.wo[i][j] or 0; local w2 = b.wo[i][j] or 0
                local alpha = util.rand(0,1)
                c.wo[i][j] = alpha*w1 + (1-alpha)*w2
            end
        end
        if math.random() < pcross then
            local alpha = util.rand(0,1)
            c.bo[i] = alpha*(a.bo[i] or 0) + (1-alpha)*(b.bo[i] or 0)
        end
    end
    return c
end

function NN.mutate(g,pmut,sigma)
    for i=1,g.H do
        for j=1,g.I do if math.random() < pmut then g.wh[i][j] = (g.wh[i][j] or 0) + util.randn(0,sigma) end end
        if math.random() < pmut then g.bh[i] = (g.bh[i] or 0) + util.randn(0,sigma) end
    end
    for i=1,g.O do
        for j=1,g.H do if math.random() < pmut then g.wo[i][j] = (g.wo[i][j] or 0) + util.randn(0,sigma) end end
        if math.random() < pmut then g.bo[i] = (g.bo[i] or 0) + util.randn(0,sigma) end
    end
end

local Agent = {}
Agent.__index = Agent
function Agent.new(id,species,genome)
    local a = setmetatable({},Agent)
    a.id = id
    a.species = species
    a.pos = {util.rand(0,cfg.world_w), util.rand(0,cfg.world_h)}
    a.orient = util.rand(-math.pi,math.pi)
    a.vel = {0,0}
    if species=="prey" then
        a.vel_base = cfg.vel_base_prey
        a.stamina = cfg.stamina_max_prey
        a.stamina_max = cfg.stamina_max_prey
        a.drain = cfg.drain_prey
        a.recover = cfg.recover_prey
        a.vision_base = cfg.vision_prey
    else
        a.vel_base = cfg.vel_base_pred
        a.stamina = cfg.stamina_max_pred
        a.stamina_max = cfg.stamina_max_pred
        a.drain = cfg.drain_pred
        a.recover = cfg.recover_pred
        a.vision_base = cfg.vision_pred
    end
    a.vel_actual = a.vel_base * (0.6 + 0.4 * a.stamina)
    a.vision = a.vision_base * (0.8 + 0.2 * a.stamina)
    a.health = 1.0
    a.energy = 1.0
    a.genome = genome or NN.newGenome(cfg.input_size,cfg.H,cfg.output_size)
    a.fitness = 0
    a.captures = 0
    a.survived = true
    a.last_action_time = 0
    a.intent_flag = false
    a.sprint_count = 0
    a.coord_timer = 0
    a.caught_count = 0
    a.metrics = {total_thrust=0,sprints=0,ambushes=0,embush_success=0,sacrifices=0,esquivos=0,rodeos=0}
    a.visible_list = {}
    a.flag_perseguido = 0
    a.survival_time = 0
    a.last_thrust = 0
    a.last_orient = a.orient
    return a
end

function Agent:inputs(neighbors,others,dt)
    local target_dx, target_dy, target_dist = 0,0,1
    if self.species=="pred" then
        local minD=1e9; local tx,ty=nil,nil
        for _,o in ipairs(others) do if o.species=="prey" and o.survived then
            local d,dx,dy = util.dist(self.pos[1],self.pos[2],o.pos[1],o.pos[2])
            if d<minD then minD=d; tx=dx; ty=dy end
        end end
        if tx then target_dx=tx; target_dy=ty; target_dist=minD end
    else
        local minD=1e9; local px,py=nil,nil
        for _,o in ipairs(others) do if o.species=="pred" and o.survived then
            local d,dx,dy=util.dist(self.pos[1],self.pos[2],o.pos[1],o.pos[2])
            if d<minD then minD=d; px=dx; py=dy end
        end end
        if px then target_dx = -px; target_dy = -py; target_dist = minD end
    end
    local dist_norm = util.clamp(target_dist / math.max(self.vision_base,1),0,1)
    local dxn = target_dx/(target_dist+1e-6)
    local dyn = target_dy/(target_dist+1e-6)
    local ang_to = math.atan2(target_dy,target_dx)
    local cosang = math.cos(util.normAngle(ang_to-self.orient))
    local sinang = math.sin(util.normAngle(ang_to-self.orient))
    local local_density = 0
    local stamina_neighbors = 0
    local ncount = 0
    for _,n in ipairs(neighbors) do
        local_density = local_density + 1
        stamina_neighbors = stamina_neighbors + (n.stamina or 0)
        ncount = ncount + 1
    end
    local_density = util.clamp(local_density / 10,0,1)
    local stamina_mean = ncount>0 and (stamina_neighbors/ncount) or self.stamina
    local inputs = {
        dxn,
        dyn,
        dist_norm,
        cosang,
        sinang,
        local_density,
        self.stamina,
        stamina_mean,
        self.flag_perseguido
    }
    return inputs
end

function Agent:update(dt,neighbors,others,world,sim)
    if not self.survived then return end
    local prev_orient = self.orient
    local inputs = self:inputs(neighbors,others,dt)
    local out = NN.forward(self.genome,inputs)
    local steer = out[1]
    local thrust = (out[2]+1)/2
    local intent = out[3]
    if thrust>cfg.sprint_threshold and self.stamina>0 then
        self.stamina = self.stamina - self.drain * thrust * dt
    else
        self.stamina = self.stamina + self.recover * dt
    end
    self.stamina = util.clamp(self.stamina,0,self.stamina_max)
    self.vel_actual = self.vel_base * (0.6 + 0.4 * self.stamina)
    self.vision = self.vision_base * (0.8 + 0.2 * self.stamina)
    self.orient = util.normAngle(self.orient + steer * 3 * dt)
    local speed = self.vel_actual * thrust
    local vx = math.cos(self.orient)*speed
    local vy = math.sin(self.orient)*speed
    self.pos[1] = self.pos[1] + vx*dt
    self.pos[2] = self.pos[2] + vy*dt
    if self.pos[1]<0 then self.pos[1]=self.pos[1]+cfg.world_w end
    if self.pos[1]>cfg.world_w then self.pos[1]=self.pos[1]-cfg.world_w end
    if self.pos[2]<0 then self.pos[2]=self.pos[2]+cfg.world_h end
    if self.pos[2]>cfg.world_h then self.pos[2]=self.pos[2]-cfg.world_h end
    self.last_action_time = self.last_action_time + dt
    local ang_delta = math.abs(util.normAngle(self.orient - prev_orient))
    if thrust>cfg.sprint_threshold then self.metrics.total_thrust = self.metrics.total_thrust + thrust end
    if self.species=="pred" then
        if intent>0.8 then
            self.intent_flag = true
            self.coord_timer = sim.time
        end
        self.visible_list = {}
        for _,o in ipairs(others) do if o.species=="prey" and o.survived then
            local d = util.dist(self.pos[1],self.pos[2],o.pos[1],o.pos[2])
            if d <= self.vision then table.insert(self.visible_list,o) end
        end end
        if #self.visible_list>0 and thrust>cfg.sprint_threshold and ang_delta>0.8 then
            self.metrics.rodeos = (self.metrics.rodeos or 0) + 1
        end
    else
        if intent>0.8 then
            self.intent_flag = true
            self.metrics.sacrifices = self.metrics.sacrifices + 1
            self.stamina = util.clamp(self.stamina - 0.4,0,self.stamina_max)
        end
        self.flag_perseguido = 0
        for _,o in ipairs(others) do if o.species=="pred" and o.survived then
            local d = util.dist(self.pos[1],self.pos[2],o.pos[1],o.pos[2])
            if d < self.vision*1.2 then self.flag_perseguido = 1; break end
        end end
        if self.flag_perseguido==1 and thrust>cfg.sprint_threshold and ang_delta>0.8 then
            self.metrics.esquivos = (self.metrics.esquivos or 0) + 1
        end
    end
    if self.survived then self.survival_time = (self.survival_time or 0) + dt end
    self.last_thrust = thrust
    self.last_orient = self.orient
end

local Sim = {}
function Sim.new()
    local s = {}
    s.time = 0
    s.gen = 1
    s.agents = {}
    s.prey = {}
    s.pred = {}
    s.running = true
    s.episode_time = cfg.episode_time
    s.generation = {prey=1,pred=1}
    s.best_fitness = {prey=0,pred=0}
    s.capture_count = 0
    s.coord_captures = 0
    s.hist = {prey={},pred={}}
    s.metrics = {sprint_rate=0,avg_stamina=0}
    s.speedMultiplier = 1
    return setmetatable(s,{__index=Sim})
end

function Sim:initPop(nprey,npred)
    self.agents = {}
    self.prey = {}
    self.pred = {}
    local id=1
    for i=1,nprey do
        local g = NN.newGenome(cfg.input_size,cfg.H,cfg.output_size)
        local a = Agent.new(id,"prey",g)
        table.insert(self.agents,a); table.insert(self.prey,a); id=id+1
    end
    for i=1,npred do
        local g = NN.newGenome(cfg.input_size,cfg.H,cfg.output_size)
        local a = Agent.new(id,"pred",g)
        table.insert(self.agents,a); table.insert(self.pred,a); id=id+1
    end
end

function Sim:resetEpisode()
    self.time = 0
    self.capture_count = 0
    self.coord_captures = 0
    for _,a in ipairs(self.agents) do
        a.pos = {util.rand(0,cfg.world_w), util.rand(0,cfg.world_h)}
        a.orient = util.rand(-math.pi,math.pi)
        a.stamina = a.stamina_max
        a.vel_actual = a.vel_base
        a.vision = a.vision_base * (0.8 + 0.2 * a.stamina)
        a.health = 1.0
        a.survived = true
        a.captures = 0
        a.fitness = 0
        a.metrics = {total_thrust=0,sprints=0,ambushes=0,embush_success=0,sacrifices=0,esquivos=0,rodeos=0}
        a.intent_flag = false
        a.last_action_time = 0
        a.survival_time = 0
        a.last_thrust = 0
        a.last_orient = a.orient
    end
end

function Sim:update(dt)
    if not self.running then return end
    self.time = self.time + dt
    for _,a in ipairs(self.agents) do a.flag_perseguido = 0 end
    for _,a in ipairs(self.prey) do if a.survived then
        for _,p in ipairs(self.pred) do if p.survived then
            local d = util.dist(a.pos[1],a.pos[2],p.pos[1],p.pos[2])
            if d < (p.vision or p.vision_base)*1.2 then a.flag_perseguido = 1; break end
        end end
    end end
    for _,a in ipairs(self.agents) do
        local neighbors = {}
        for _,b in ipairs(self.agents) do
            if a~=b and util.dist(a.pos[1],a.pos[2],b.pos[1],b.pos[2]) < ((a.vision or a.vision_base) or 100) then table.insert(neighbors,b) end
        end
        a:update(dt,neighbors,self.agents,{w=cfg.world_w,h=cfg.world_h},self)
    end
    for _,p in ipairs(self.pred) do
        if p.survived then
            for i=#self.prey,1,-1 do
                local q = self.prey[i]
                if q.survived then
                    local d = util.dist(p.pos[1],p.pos[2],q.pos[1],q.pos[2])
                    if d < cfg.capture_radius then
                        q.survived = false
                        p.captures = p.captures + 1
                        self.capture_count = self.capture_count + 1
                        p.fitness = p.fitness + 10
                        if self:checkCoordCapture(p) then
                            p.fitness = p.fitness + 2
                            p.metrics.embush_success = p.metrics.embush_success + 1
                            self.coord_captures = self.coord_captures + 1
                        end
                    end
                end
            end
        end
    end
    if self.time >= self.episode_time then
        self:endEpisode()
    end
    if #self.prey == self.capture_count then
        self:endEpisode()
    end
    local total_st, total_agents = 0,0
    for _,a in ipairs(self.agents) do total_st = total_st + (a.stamina or 0); total_agents = total_agents + 1 end
    self.metrics.avg_stamina = total_agents>0 and (total_st/total_agents) or 0
end

function Sim:checkCoordCapture(agent)
    local t = agent.coord_timer or 0
    local cnt=0
    local partners=0
    for _,p in ipairs(self.pred) do
        local d = util.dist(agent.pos[1],agent.pos[2],p.pos[1],p.pos[2])
        if d <= cfg.comm_range then
            partners = partners + 1
            if p.intent_flag and math.abs((p.coord_timer or 0)-t) <= cfg.coord_window then cnt = cnt + 1 end
        end
    end
    return cnt >= math.max(1,math.floor(partners/2))
end

function Sim:endEpisode()
    for _,a in ipairs(self.agents) do
        if a.species=="prey" then
            a.fitness = a.fitness + (a.survived and 5 or 0)
            local mean_surv = 0
            local n=0
            for _,nbor in ipairs(self.prey) do if nbor~=a then mean_surv = mean_surv + (nbor.survived and 1 or 0); n=n+1 end end
            a.fitness = a.fitness + 0.5 * (n>0 and (mean_surv/n* (self.episode_time)) or 0)
            a.fitness = a.fitness - 0.02 * (a.metrics.total_thrust or 0)
        else
            a.fitness = a.fitness + 10 * (a.captures or 0) - 0.01 * (self.episode_time - (a.captures>0 and self.episode_time or 0))
        end
    end
    self:runGA()
    self.time = 0
    self.gen = self.gen + 1
    self.generation = {prey=self.generation.prey+1,pred=self.generation.pred+1}
    self:resetEpisode()
end

function Sim:selectChampionPrey()
    local best=nil
    local bestScore = -1/0
    for _,a in ipairs(self.prey) do
        local score = (a.survival_time or 0) * cfg.prey_survival_weight + (a.metrics.esquivos or 0) * cfg.prey_esquivo_weight
        if score > bestScore then bestScore = score; best = a end
    end
    return best
end

function Sim:selectChampionPred()
    local best=nil
    local bestScore = -1/0
    for _,a in ipairs(self.pred) do
        local score = (a.captures or 0) * cfg.pred_capture_weight + (a.metrics.rodeos or 0) * cfg.pred_rodeo_weight
        if score > bestScore then bestScore = score; best = a end
    end
    return best
end

local function is_array(t)
    if type(t)~="table" then return false end
    local i=0
    for k in pairs(t) do
        if type(k)~="number" then return false end
        i=i+1
    end
    for j=1,i do if t[j]==nil then return false end end
    return true
end

local function serialize(o)
    local t = type(o)
    if t=="number" or t=="boolean" then return tostring(o) end
    if t=="string" then return string.format("%q",o) end
    if t=="table" then
        local s = "{"
        if is_array(o) then
            for i=1,#o do s = s .. serialize(o[i]) .. "," end
        else
            for k,v in pairs(o) do s = s .. "["..serialize(k).."]="..serialize(v).."," end
        end
        s = s .. "}"
        return s
    end
    return "nil"
end

function Sim:saveState(filename)
    local state = {gen=self.gen,prey_genomes={},pred_genomes={}}
    for _,a in ipairs(self.prey) do table.insert(state.prey_genomes,a.genome) end
    for _,a in ipairs(self.pred) do table.insert(state.pred_genomes,a.genome) end
    local chunk = "return "..serialize(state)
    love.filesystem.write(filename or "sim_state.lua",chunk)
end

function Sim:loadState(filename)
    local file = filename or "sim_state.lua"
    if not love.filesystem.getInfo(file) then return false end
    local chunk,err = love.filesystem.load(file)
    if not chunk then return false end
    local ok,state = pcall(chunk)
    if not ok or not state then return false end
    local loaded_preys = state.prey_genomes or {}
    local loaded_preds = state.pred_genomes or {}
    local function pickGenome(list,i)
        if #list==0 then return nil end
        if i <= #list then return util.copy(list[i]) end
        return util.copy(list[((i-1) % #list) + 1])
    end
    self.prey = {}
    self.pred = {}
    self.agents = {}
    local newid=1
    for i=1,math.max(0,cfg.prey_count) do
        local g = pickGenome(loaded_preys,i) or NN.newGenome(cfg.input_size,cfg.H,cfg.output_size)
        local a = Agent.new(newid,"prey",g); newid=newid+1
        table.insert(self.prey,a); table.insert(self.agents,a)
    end
    for i=1,math.max(0,cfg.pred_count) do
        local g = pickGenome(loaded_preds,i) or NN.newGenome(cfg.input_size,cfg.H,cfg.output_size)
        local a = Agent.new(newid,"pred",g); newid=newid+1
        table.insert(self.pred,a); table.insert(self.agents,a)
    end
    self.gen = state.gen or self.gen
    self:resetEpisode()
    return true
end

function Sim:runGA()
    local preyChampion = self:selectChampionPrey()
    local predChampion = self:selectChampionPred()
    local newprey = {}
    local newpred = {}
    if preyChampion then
        for i=1,math.max(1,cfg.prey_count) do
            local g = NN.clone(preyChampion.genome)
            if cfg.pmut>0 then NN.mutate(g,cfg.pmut,0.1) end
            table.insert(newprey,g)
        end
    else
        for i=1,cfg.prey_count do table.insert(newprey, NN.newGenome(cfg.input_size,cfg.H,cfg.output_size)) end
    end
    if predChampion then
        for i=1,math.max(1,cfg.pred_count) do
            local g = NN.clone(predChampion.genome)
            if cfg.pmut>0 then NN.mutate(g,cfg.pmut,0.1) end
            table.insert(newpred,g)
        end
    else
        for i=1,cfg.pred_count do table.insert(newpred, NN.newGenome(cfg.input_size,cfg.H,cfg.output_size)) end
    end
    local newid=1
    local newagents={}
    self.prey={}
    self.pred={}
    for i=1,#newprey do
        local a = Agent.new(newid,"prey",newprey[i]); newid=newid+1
        table.insert(newagents,a); table.insert(self.prey,a)
    end
    for i=1,#newpred do
        local a = Agent.new(newid,"pred",newpred[i]); newid=newid+1
        table.insert(newagents,a); table.insert(self.pred,a)
    end
    self.agents = newagents
end

function Sim:runGenerationsFast(n, step_dt)
    step_dt = step_dt or 0.5
    local target = self.gen + n
    local prev_running = self.running
    self.running = true
    while self.gen < target do
        self:update(step_dt)
    end
    self.running = prev_running
end

local sim = Sim.new()

function love.load()
    math.randomseed(os.time())
    love.window.setMode(cfg.world_w, cfg.world_h+200,{resizable=false})
    sim:initPop(cfg.prey_count,cfg.pred_count)
    sim:resetEpisode()
    fonts = {small=love.graphics.newFont(12),big=love.graphics.newFont(14)}
    ui = {showVision=false,debug=false,mutationEnabled=true}
end

function love.update(dt)
    if love.keyboard.isDown("p") then sim.running = not sim.running end
    if sim.speedMultiplier <= 1 then
        if sim.running then sim:update(dt) end
    else
        local steps = sim.speedMultiplier
        local step_dt = dt
        for i=1,steps do
            if not sim.running then break end
            sim:update(step_dt)
        end
    end
end

function drawAgent(a)
    if a.species=="prey" then
        love.graphics.setColor(0.2,0.7,0.2)
        love.graphics.circle("fill",a.pos[1],a.pos[2],cfg.prey_radius)
        love.graphics.setColor(0,0,0)
        love.graphics.rectangle("fill",a.pos[1]-10,a.pos[2]-12,20,4)
        love.graphics.setColor(0.8,0.2,0.2)
        love.graphics.rectangle("fill",a.pos[1]-10,a.pos[2]-12,20*(a.stamina/a.stamina_max),4)
    else
        love.graphics.setColor(0.7,0.2,0.2)
        love.graphics.rectangle("fill",a.pos[1]-cfg.pred_size/2,a.pos[2]-cfg.pred_size/2,cfg.pred_size,cfg.pred_size)
        love.graphics.setColor(0,0,0)
        love.graphics.rectangle("fill",a.pos[1]-12,a.pos[2]-18,24,5)
        love.graphics.setColor(0.2,0.6,0.9)
        love.graphics.rectangle("fill",a.pos[1]-12,a.pos[2]-18,24*(a.stamina/a.stamina_max),5)
    end
    if ui.showVision then
        love.graphics.setColor(1,1,1,0.08)
        love.graphics.circle("line",a.pos[1],a.pos[2],a.vision or 50)
    end
    if ui.debug then
        love.graphics.setColor(1,1,1)
        love.graphics.print(string.format("s:%.2f e:%.0f r:%.0f",a.stamina,(a.metrics.esquivos or 0),(a.metrics.rodeos or 0)),a.pos[1]+8,a.pos[2]+8)
    end
end

function love.draw()
    love.graphics.clear(0.09,0.09,0.12)
    love.graphics.setFont(fonts.big)
    for _,a in ipairs(sim.agents) do if a.survived then drawAgent(a) end end
    love.graphics.setColor(1,1,1)
    love.graphics.rectangle("fill",0,cfg.world_h,cfg.world_w,200)
    love.graphics.setColor(0,0,0)
    love.graphics.setFont(fonts.small)
    local info = {
        ("Gen: %d  Tiempo: %.1f / %.1f  Presa: %d  Depredador: %d  Capturados: %d  CoordCaptura: %d"):format(sim.gen,sim.time,sim.episode_time,#(sim.prey or {}),#(sim.pred or {}),sim.capture_count,sim.coord_captures),
        ("Velocidad x%d  Stamina: %.2f  PMut: %.3f  PCross: %.2f  Sprint Thr: %.2f"):format(sim.speedMultiplier,sim.metrics.avg_stamina,cfg.pmut,cfg.pcross,cfg.sprint_threshold),
        ("Controles: +/- presa   ]/[ depredador   m mutación   p pausa   g finalizar  s save   l load   v radio de vision   d desarrollo   ,/. velocidad simulacion   o salto de 10 gens")
    }
    for i,line in ipairs(info) do love.graphics.print(line,10,cfg.world_h+10+14*(i-1)) end
    love.graphics.setColor(0,0,0)
    love.graphics.print("Observables:",10,cfg.world_h+60)
    local obsx = 110
    love.graphics.print(("Best prey fitness: %.2f   Best pred fitness: %.2f"):format(sim.best_fitness.prey,sim.best_fitness.pred),obsx,cfg.world_h+60)
    love.graphics.print(("Sprints rate: %.2f   Stamina mean: %.2f"):format(sim.metrics.sprint_rate,sim.metrics.avg_stamina),obsx,cfg.world_h+80)
    drawHistogram(10,cfg.world_h+100,400,80)
end

function drawHistogram(x,y,w,h)
    local prey_f = {}
    for _,a in ipairs(sim.prey) do table.insert(prey_f,a.fitness or 0) end
    local pred_f = {}
    for _,a in ipairs(sim.pred) do table.insert(pred_f,a.fitness or 0) end
    love.graphics.setColor(0.9,0.9,0.9)
    love.graphics.rectangle("line",x,y,w,h)
    local bins = cfg.hist_bins
    local maxv = 1
    for _,v in ipairs(prey_f) do if v>maxv then maxv=v end end
    for _,v in ipairs(pred_f) do if v>maxv then maxv=v end end
    for i=1,bins do
        local bx = x + (i-1)*(w/bins)
        local bw = w/bins - 1
        local pc = 0
        for _,v in ipairs(prey_f) do
            local bi = math.min(bins,math.max(1,math.floor((v/maxv)*bins)+1))
            if bi==i then pc = pc + 1 end
        end
        local pd = 0
        for _,v in ipairs(pred_f) do
            local bi = math.min(bins,math.max(1,math.floor((v/maxv)*bins)+1))
            if bi==i then pd = pd + 1 end
        end
        local ph = (pc/(#prey_f+1e-6))*h
        local dh = (pd/(#pred_f+1e-6))*h
        love.graphics.setColor(0.2,0.7,0.2,0.6)
        love.graphics.rectangle("fill",bx,y+h-ph,bw,ph)
        love.graphics.setColor(0.7,0.2,0.2,0.6)
        love.graphics.rectangle("fill",bx,y+h-dh,bw,dh/2)
    end
end

function love.keypressed(k)
    if k=="+" or k=="=" then
        cfg.prey_count = cfg.prey_count + 5
        for i=1,5 do local a=Agent.new(#sim.agents+1,"prey"); table.insert(sim.agents,a); table.insert(sim.prey,a) end
    elseif k=="-" then
        for i=1,5 do
            if #sim.prey>0 then local a=table.remove(sim.prey); for j=#sim.agents,1,-1 do if sim.agents[j]==a then table.remove(sim.agents,j); break end end end
        end
        cfg.prey_count = math.max(0,cfg.prey_count-5)
    elseif k=="]" then
        cfg.pred_count = cfg.pred_count + 2
        for i=1,2 do local a=Agent.new(#sim.agents+1,"pred"); table.insert(sim.agents,a); table.insert(sim.pred,a) end
    elseif k=="[" then
        for i=1,2 do
            if #sim.pred>0 then local a=table.remove(sim.pred); for j=#sim.agents,1,-1 do if sim.agents[j]==a then table.remove(sim.agents,j); break end end end
        end
        cfg.pred_count = math.max(0,cfg.pred_count-2)
    elseif k=="m" then
        ui.mutationEnabled = not ui.mutationEnabled
        cfg.pmut = ui.mutationEnabled and cfg.pmut or 0
    elseif k=="p" then
        sim.running = not sim.running
    elseif k=="g" then
        sim:endEpisode()
    elseif k=="s" then
        sim:saveState("sim_state.lua")
    elseif k=="l" then
        local ok = sim:loadState("sim_state.lua")
        if not ok then print("No state file or load failed") end
    elseif k=="v" then
        ui.showVision = not ui.showVision
    elseif k=="d" then
        ui.debug = not ui.debug
    elseif k=="," then
        sim.speedMultiplier = math.max(1,math.floor(sim.speedMultiplier/2))
    elseif k=="." then
        sim.speedMultiplier = math.min(1024,sim.speedMultiplier*2)
    elseif k=="o" then
        sim:runGenerationsFast(10,0.5)
    end
end