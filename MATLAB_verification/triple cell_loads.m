%% Ground Structure Optimization using GA and Linear Programming
% This script optimizes a ground structure problem using a Genetic Algorithm (GA)
% for node displacement optimization and Linear Programming (LP) for material usage minimization.

clear; close all; clc;

%% Parameters
% Optimization parameters
kappa = 1.0; % Weight factor for material usage
ColTol = 0.999999; % Tolerance for collinearity of bars
Cutoff = 0.0; % Minimum cross-sectional area for plotting
Ng = 50; % Number of groups for color coding bars in plots
TolX = 0; % Tolerance for node movement in x
TolY = 0; % Tolerance for node movement in y
RestrictDomain = []; % No restriction for box domain

% Define the nodes where loads are applied and their respective magnitudes
% Updated for 7 nodes wide and 3 nodes tall structure with 10 kN loads
LoadNodes = [3, 6, 9, 12, 15, 18, 21];
LoadMagnitudes = [10, 10, 10, 10, 10, 10, 10];

% Generate the structured orthogonal domain
% Updated Nx=6 (7 nodes wide), Ny=2 (3 nodes tall), Lx=6, Ly=2
[NODE, ELEM, SUPP, LOAD, BARS] = StructDomain(6, 2, 6, 2, 'bridge', LoadNodes, LoadMagnitudes);

% Visualize node numbering
figure;
visualizeNodes(NODE, SUPP, LOAD);
title('Node Numbering');

% Define movable nodes for optimization
% Updated movable nodes based on new structure
movableNodesX = [5, 11, 17];
movableNodesY = [3, 6, 9, 12, 15, 18, 21];

%% Optimization with GA
numVarsX = length(movableNodesX); % Number of variables to optimize (x-coordinates of movable nodes)
numVarsY = length(movableNodesY); % Number of variables to optimize (y-coordinates of movable nodes)
numVars = numVarsX + numVarsY; % Total number of variables

% Combine lower and upper bounds for x and y coordinates
lb = [NODE(movableNodesX, 1) - TolX; NODE(movableNodesY, 2) - TolY]; % Lower bounds for x and y coordinates
ub = [NODE(movableNodesX, 1) + TolX; NODE(movableNodesY, 2) + TolY]; % Upper bounds for x and y coordinates

% Set GA options
options = optimoptions('ga', ...
    'PopulationSize', 50, ...
    'MaxGenerations', 100, ... 
    'Display', 'iter', ...
    'PlotFcn', {@gaplotbestf, @gaplotbestindiv}, ... % Simplified Plot Functions for Clarity
    'UseParallel', false);

% Define the fitness function for the GA
fitnessFcn = @(xy) evaluateStructure(xy, movableNodesX, movableNodesY, NODE, ELEM, SUPP, LOAD, BARS, ColTol, kappa);

% Initialize the global variable for total volumes
global totalVolumes;
totalVolumes = [];

% Run the genetic algorithm to find the best node positions
[bestXY, bestVol] = ga(fitnessFcn, numVars, [], [], [], [], lb, ub, [], options);

% Update the NODE coordinates with the best solution found
NODE(movableNodesX, 1) = bestXY(1:numVarsX);
NODE(movableNodesY, 2) = bestXY(numVarsX+1:end);

%% Plotting Total Volume vs. Generations
% Extract and plot total volume vs. generations
figure;
global totalVolumes;
plot(1:length(totalVolumes), totalVolumes, '-o');
xlabel('Generations');
ylabel('Total Volume');
title('Total Volume vs. Generations');

%% Ground Structure Method
% Plot the base mesh
figure;
PlotPolyMesh(NODE, ELEM, SUPP, LOAD);
title('Ground Structure');

% Use the initial bars (BARS) directly from the structured domain
Nn = size(NODE, 1); % Number of nodes
Ne = length(ELEM); % Number of elements
Nb = size(BARS, 1); % Number of bars

% Get reaction nodes (supports)
[BC] = GetSupports(SUPP);

% Get equilibrium matrix and member lengths
[BT, L] = GetMatrixBT(NODE, BARS, BC, Nn, Nb);

% Get nodal force vector
[F] = GetVectorF(LOAD, BC, Nn);

fprintf('Mesh: Elements %d, Nodes %d, Bars %d\n', Ne, Nn, Nb);

% Prepare matrix for linear programming
BTBT = [BT -BT]; 
LL = [L; kappa * L]; 
sizeBTBT = whos('BTBT'); 
clear BT L;
fprintf('Matrix [BT -BT]: %d x %d in %gMB (%gGB full)\n', length(F), length(LL), sizeBTBT.bytes/2^20, 16 * (2 * Nn) * Nb / 2^30);

% Solve the linear programming problem to find optimal cross-sectional areas
tic;
[S, vol, exitflag] = linprog(LL, [], [], BTBT, F, zeros(2 * Nb, 1));
fprintf('Objective V = %f\nlinprog CPU time = %g s\n', vol, toc);

% Check if LP was successful
if exitflag ~= 1
    error('Linear programming did not converge to a solution.');
end

% Separate slack variables and compute final areas and member forces
S = reshape(S, numel(S) / 2, 2); % Separate slack variables
A = S(:, 1) + kappa * S(:, 2); % Get cross-sectional areas
N = S(:, 1) - S(:, 2); % Get member forces

%% Plotting Optimized Structure
% Plot the final ground structure with cross-sectional areas
figure;
memberInfo = PlotGroundStructure(NODE, BARS, A, Cutoff, Ng, SUPP);
title('Optimised Structure');

% Print member information
fprintf('\nMember Information:\n');
fprintf('--------------------\n');
for i = 1:length(memberInfo)
    fprintf('Member %d: Start(%.2f, %.2f), End(%.2f, %.2f), Cross-Area: %.2f\n', ...
        i, memberInfo{i}.Start(1), memberInfo{i}.Start(2), ...
        memberInfo{i}.End(1), memberInfo{i}.End(2), ...
        memberInfo{i}.CrossArea);
end

% **Added Section: Print Member Forces**
fprintf('\nMember Forces in Optimal Structure:\n');
fprintf('------------------------------------\n');
for i = 1:length(A)
    fprintf('Member %d:\n', i);
    fprintf('  Start Node: %d (%.2f, %.2f)\n', BARS(i,1), NODE(BARS(i,1),1), NODE(BARS(i,1),2));
    fprintf('  End Node:   %d (%.2f, %.2f)\n', BARS(i,2), NODE(BARS(i,2),1), NODE(BARS(i,2),2));
    fprintf('  Force:      %.2f kN\n\n', N(i));
end

% Calculate and print angles at each node along with member cross-sectional areas
anglesAtNodes = calculateAndPrintAngles(NODE, memberInfo);

% Set the angle tolerance
angleTolerance = 1.0; % Tolerance of 1.0 degrees
areaTolerance = 0.2; % Tolerance of 0.2 for cross-sectional areas

% Count occurrences of different angle sets with angle tolerance
angleSetCounts = countAngleSetOccurrencesWithTolerance(anglesAtNodes, angleTolerance);

% Print angle set occurrences
fprintf('\nAngle Set Counts:\n');
angleSets = keys(angleSetCounts);
for i = 1:length(angleSets)
    angleSet = angleSets{i};
    count = angleSetCounts(angleSet);
    fprintf('%s: %d\n', angleSet, count);
end
fprintf('\nTotal Different Types of Nodes: %d\n', length(angleSetCounts));

% Count occurrences of different angle sets including cross-sectional areas with tolerances
angleSetCountsIncludingAreas = countAngleSetOccurrencesIncludingAreasWithTolerance(anglesAtNodes, angleTolerance, areaTolerance);

% Print angle set occurrences including cross-sectional areas
fprintf('\nAngle Set Counts Including Cross-Sectional Areas:\n');
angleSetsIncludingAreas = keys(angleSetCountsIncludingAreas);
for i = 1:length(angleSetsIncludingAreas)
    angleSet = angleSetsIncludingAreas{i};
    count = angleSetCountsIncludingAreas(angleSet);
    fprintf('%s: %d\n', angleSet, count);
end
fprintf('\nTotal Different Types of Nodes Including Cross-Sectional Areas: %d\n', length(angleSetCountsIncludingAreas));

%% Function Definitions

% Function to visualize node numbering with improved readability and comments
function visualizeNodes(NODE, SUPP, LOAD)
    figure;
    hold on;
    axis equal;
    axis off;
    plot(NODE(:, 1), NODE(:, 2), 'ko'); 
    text(NODE(:, 1) + 0.1, NODE(:, 2), arrayfun(@num2str, (1:size(NODE, 1))', 'UniformOutput', false), 'FontSize', 8); 
    title('Node Numbering');

    % Highlight movable nodes in red for x and magenta for y
    movableNodesX = [5, 11, 17];
    movableNodesY = [3, 6, 9, 12, 15, 18, 21];
    plot(NODE(movableNodesX, 1), NODE(movableNodesX, 2), 'ro', 'MarkerSize', 10, 'DisplayName', 'Movable X'); 
    plot(NODE(movableNodesY, 1), NODE(movableNodesY, 2), 'mo', 'MarkerSize', 10, 'DisplayName', 'Movable Y'); 

    % Highlight nodes movable in both x and y directions in yellow
    movableNodesXY = intersect(movableNodesX, movableNodesY);
    if ~isempty(movableNodesXY)
        plot(NODE(movableNodesXY, 1), NODE(movableNodesXY, 2), 'yo', 'MarkerSize', 10, 'DisplayName', 'Movable XY'); 
    end

    % Highlight support nodes in blue
    supportNodes = SUPP(:, 1);
    plot(NODE(supportNodes, 1), NODE(supportNodes, 2), 'bo', 'MarkerSize', 10, 'DisplayName', 'Supports'); 

    % Highlight load nodes in green
    loadNodes = LOAD(:, 1);
    plot(NODE(loadNodes, 1), NODE(loadNodes, 2), 'go', 'MarkerSize', 10, 'DisplayName', 'Loads'); 

    legend('Location', 'bestoutside');
    hold off;
end

% Custom Plot Function for GA with modularization and consistent tolerances
function state = gaPlotFunction(options, state, flag, MovableNodeIndicesX, MovableNodeIndicesY, NODE, ELEM, SUPP, LOAD, BARS, ColTol, kappa, Cutoff, Ng)
    global totalVolumes;
    persistent figHandle;
    if isempty(figHandle) || ~isvalid(figHandle)
        figHandle = figure;
        totalVolumes = []; % Initialize totalVolumes
    end
    % Get the current best solution
    [~, idx] = min(state.Score);
    bestXY = state.Population(idx, :);

    % Update the NODE coordinates with the current best solution
    NODE(MovableNodeIndicesX, 1) = bestXY(1:length(MovableNodeIndicesX));
    NODE(MovableNodeIndicesY, 2) = bestXY(length(MovableNodeIndicesX)+1:end);

    % Solve the linear programming problem
    [vol, exitflag, A] = solveLP(NODE, BARS, SUPP, LOAD, kappa);
    if exitflag ~= 1
        return;
    end

    % Store the volume for the current generation
    totalVolumes = [totalVolumes, vol];

    % Update the plot
    figure(figHandle);
    clf;
    PlotGroundStructure(NODE, BARS, A, Cutoff, Ng, SUPP);
    title(sprintf('Optimisation Process: Generation %d', state.Generation));
    drawnow;

    % Calculate and print angle sets and counts
    memberInfo = PlotGroundStructure(NODE, BARS, A, Cutoff, Ng, SUPP);
    anglesAtNodes = calculateAndPrintAngles(NODE, memberInfo);

    % Set the angle tolerance
    angleTolerance = 1.0; % Tolerance of 1.0 degrees
    areaTolerance = 0.2; % Tolerance of 0.2 for cross-sectional areas

    % Count occurrences of different angle sets with angle tolerance
    angleSetCounts = countAngleSetOccurrencesWithTolerance(anglesAtNodes, angleTolerance);

    % Print angle set occurrences
    fprintf('\nAngle Set Counts (Generation %d):\n', state.Generation);
    angleSets = keys(angleSetCounts);
    for i = 1:length(angleSets)
        angleSet = angleSets{i};
        count = angleSetCounts(angleSet);
        fprintf('%s: %d\n', angleSet, count);
    end
    fprintf('\nTotal Different Types of Nodes: %d\n', length(angleSetCounts));

    % Count occurrences of different angle sets including cross-sectional areas with tolerances
    angleSetCountsIncludingAreas = countAngleSetOccurrencesIncludingAreasWithTolerance(anglesAtNodes, angleTolerance, areaTolerance);

    % Print angle set occurrences including cross-sectional areas
    fprintf('\nAngle Set Counts Including Cross-Sectional Areas (Generation %d):\n', state.Generation);
    angleSetsIncludingAreas = keys(angleSetCountsIncludingAreas);
    for i = 1:length(angleSetsIncludingAreas)
        angleSet = angleSetsIncludingAreas{i};
        count = angleSetCountsIncludingAreas(angleSet);
        fprintf('%s: %d\n', angleSet, count);
    end
    fprintf('\nTotal Different Types of Nodes Including Cross-Sectional Areas: %d\n', length(angleSetCountsIncludingAreas));
end

% Function to solve the linear programming problem with error handling
function [vol, exitflag, A] = solveLP(NODE, BARS, SUPP, LOAD, kappa)
    try
        Nn = size(NODE, 1); % Number of nodes
        Nb = size(BARS, 1); % Number of bars

        % Get reaction nodes (supports)
        [BC] = GetSupports(SUPP);

        % Get equilibrium matrix and member lengths
        [BT, L] = GetMatrixBT(NODE, BARS, BC, Nn, Nb);

        % Get nodal force vector
        [F] = GetVectorF(LOAD, BC, Nn);

        % Prepare matrix for linear programming
        BTBT = [BT -BT];
        LL = [L; kappa * L];

        % Solve the linear programming problem to find optimal cross-sectional areas
        [S, vol, exitflag] = linprog(LL, [], [], BTBT, F, zeros(2 * Nb, 1));
        if exitflag == 1
            % Separate slack variables and compute final areas
            S = reshape(S, numel(S) / 2, 2); % Separate slack variables
            A = S(:, 1) + kappa * S(:, 2); % Get cross-sectional areas
        else
            A = [];
        end
    catch ME
        warning('Error occurred in solveLP: %s', ME.message);
        vol = inf;
        exitflag = -1;
        A = [];
    end
end

% Function to evaluate the structure
function score = evaluateStructure(xy, MovableNodeIndicesX, MovableNodeIndicesY, NODE, ELEM, SUPP, LOAD, BARS, ColTol, kappa)
    % Update NODE coordinates with GA variables
    NODE(MovableNodeIndicesX, 1) = xy(1:length(MovableNodeIndicesX));
    NODE(MovableNodeIndicesY, 2) = xy(length(MovableNodeIndicesX)+1:end);
    
    % Solve the linear programming problem
    [vol, exitflag, ~] = solveLP(NODE, BARS, SUPP, LOAD, kappa);
    if exitflag ~= 1
        vol = inf; % Penalize infeasible solutions
    end
    score = vol; % The objective to minimize is the volume
end

% Function to generate the structured domain
function [NODE, ELEM, SUPP, LOAD, BARS] = StructDomain(Nx, Ny, Lx, Ly, ProblemID, LoadNodes, LoadMagnitudes)
    % Generate structured-orthogonal domains with alternating diagonals
    [X, Y] = meshgrid(linspace(0, Lx, Nx+1), linspace(0, Ly, Ny+1));
    NODE = [reshape(X, numel(X), 1) reshape(Y, numel(Y), 1)];
    k = 0; ELEM = cell(2*Nx*Ny, 1); % Increase size to accommodate diagonal elements
    for j = 1:Ny
        for i = 1:Nx
            k = k + 1;
            n1 = (i-1)*(Ny+1) + j; 
            n2 = i*(Ny+1) + j;
            n3 = n2 + 1;
            n4 = n1 + 1;
            ELEM{k} = [n1 n2 n3 n4]; % Add rectangular element

            % Add alternating diagonal elements
            k = k + 1;
            if mod(i+j, 2) == 0
                ELEM{k} = [n1 n3]; % Diagonal from bottom-left to top-right
            else
                ELEM{k} = [n2 n4]; % Diagonal from bottom-right to top-left
            end
        end
    end

    % Define support conditions specifically for a bridge
    SUPP = [1 1 1; (Nx)*(Ny+1)+1 1 1]; % Nodes 1 and Nx*(Ny+1)+1 are supports
    
    % Apply downward loads to the specified nodes with given magnitudes
    LOAD = arrayfun(@(n, m) [n, 0, -m], LoadNodes, LoadMagnitudes, 'UniformOutput', false);
    LOAD = vertcat(LOAD{:}); % Convert cell array to matrix

    % Generate bars (connections) from elements
    BARS = [];
    for i = 1:length(ELEM)
        elem = ELEM{i};
        if length(elem) == 2 % It's a bar
            BARS = [BARS; elem];
        elseif length(elem) == 4 % It's a rectangle
            BARS = [BARS; elem(1) elem(2); elem(2) elem(3); elem(3) elem(4); elem(4) elem(1)];
        end
    end
end

% Function to plot the mesh
function [] = PlotPolyMesh(NODE, ELEM, SUPP, LOAD)
    hold on; axis equal; axis off;
    MaxNVer = max(cellfun(@numel, ELEM)); % Maximum number of vertices in mesh
    PadWNaN = @(E) [E NaN(1, MaxNVer - numel(E))]; % Pad cells with NaN
    ElemMat = cellfun(PadWNaN, ELEM, 'UniformOutput', false);
    ElemMat = vertcat(ElemMat{:}); % Create padded element matrix
    patch('Faces', ElemMat, 'Vertices', NODE, 'FaceColor', 'w'); % Plot mesh elements

    if (nargin >= 4 && ~isempty(SUPP) && ~isempty(LOAD))
        plot(NODE(SUPP(:, 1), 1), NODE(SUPP(:, 1), 2), 'b>', 'MarkerSize', 8, 'DisplayName', 'Supports'); % Plot supports
        plot(NODE(LOAD(:, 1), 1), NODE(LOAD(:, 1), 2), 'm^', 'MarkerSize', 8, 'DisplayName', 'Loads'); % Plot loads
        legend('Location', 'bestoutside');
    end
    axis tight; drawnow;
end

% Function to get support boundary conditions
function [BC] = GetSupports(SUPP)
    % Return degrees-of-freedom with fixed (prescribed) displacements
    Nf = sum(~isnan(SUPP(:,2:3)), 'all');
    BC = zeros(Nf,1); 
    j = 0;
    for i=1:size(SUPP,1)
        if ~isnan(SUPP(i,2)), j = j + 1; BC(j) = 2*SUPP(i,1) - 1; end
        if ~isnan(SUPP(i,3)), j = j + 1; BC(j) = 2*SUPP(i,1); end
    end
    if j ~= Nf, error('Parsing number mismatch on BCs.'); end
end

% Function to get equilibrium matrix and member lengths
function [BT, L] = GetMatrixBT(NODE, BARS, BC, Nn, Nb)
    % Generate equilibrium matrix BT and get member lengths L
    D = [NODE(BARS(:,2),1)-NODE(BARS(:,1),1) NODE(BARS(:,2),2)-NODE(BARS(:,1),2)];
    L = sqrt(D(:,1).^2 + D(:,2).^2);
    D = [D(:,1)./L D(:,2)./L];
    BT = sparse([2*BARS(:,1)-1 2*BARS(:,1) 2*BARS(:,2)-1 2*BARS(:,2)],...
                repmat((1:Nb)',1,4),[-D D],2*Nn,Nb);
    BT(BC,:) = []; % Apply constraints
end

% Function to get nodal force vector
function [F] = GetVectorF(LOAD, BC, Nn)
    % Return nodal force vector
    Nl = sum(~isnan(LOAD(:,2:3)), 'all');
    F = sparse([],[],[],2*Nn,1,Nl);
    for i=1:size(LOAD,1)
        n = LOAD(i,1);
        if ~isnan(LOAD(i,2)), F(2*n-1) = LOAD(i,2); end
        if ~isnan(LOAD(i,3)), F(2*n) = LOAD(i,3); end
    end
    F(BC) = []; % Apply constraints
end

% Function to plot the final ground structure
function memberInfo = PlotGroundStructure(NODE, BARS, A, Cutoff, Ng, SUPP)
    hold on; axis equal; axis off;

    % Define a colormap for plotting
    cmap = jet(Ng);  % Create a jet colormap with Ng colors
    colorIndices = ceil(Ng * (A / max(A))); % Normalize areas to [0, Ng]
    colorIndices(colorIndices == 0) = 1; % Ensure indices start at 1

    A_normalized = A / max(A); % Normalize areas to [0,1]
    ind = find(A_normalized > Cutoff); % Find indices of bars with area above cutoff
    MyGroup = ceil(Ng * A_normalized(ind)); % Group bars by their areas
    Groups = cell(Ng, 1); % Initialize groups
    memberInfo = {}; % Initialize cell array to store member info
    for i = 1:Ng
        Groups{i} = ind(MyGroup == i); 
    end
    for i = Ng:-1:1 % Plot each group of bars
        if ~isempty(Groups{i})
            for j = 1:length(Groups{i})
                barIndex = Groups{i}(j);
                XY = [NODE(BARS(barIndex, 1), :); NODE(BARS(barIndex, 2), :)];
                GroupArea = mean(A_normalized(barIndex)); % Mean area for this group
                % Plot bars with line width proportional to their area
                plot(XY(:,1), XY(:,2), 'LineWidth', 5 * sqrt(GroupArea), 'Color', cmap(i, :))
                % Store member information
                memberInfo{end+1} = struct('Start', NODE(BARS(barIndex, 1), :), ...
                                           'End', NODE(BARS(barIndex, 2), :), ...
                                           'CrossArea', round(GroupArea * max(A), 2));
            end
        end
    end

    % Plot supports as small triangles
    for i = 1:size(SUPP, 1)
        plot(NODE(SUPP(i, 1), 1), NODE(SUPP(i, 1), 2), 'r^', 'MarkerSize', 8, 'MarkerFaceColor', 'm');
    end

    fprintf('-PLOT- Cutoff %g, Groups %g, Bars plotted %g\n', Cutoff, Ng, length(ind));

    % Add colorbar
    colorbar('Ticks', linspace(0, 1, Ng), 'TickLabels', num2cell(linspace(min(A_normalized), max(A_normalized), Ng)));
    colormap(jet(Ng));
    caxis([min(A_normalized) max(A_normalized)]);
    title(colorbar, 'Cross-sectional area');
end

% Function to calculate and print the connection angles at each node, including cross-sectional areas
function anglesAtNodes = calculateAndPrintAngles(NODE, memberInfo)
    anglesAtNodes = struct();
    angleSets = {};
    for i = 1:size(NODE, 1)
        connectedBars = [];
        for j = 1:length(memberInfo)
            if isequal(NODE(i, :), memberInfo{j}.Start) || isequal(NODE(i, :), memberInfo{j}.End)
                connectedBars(end+1) = j; % Store index of the connected bar
            end
        end
        numConnections = length(connectedBars);
        if numConnections == 1
            nodePos = NODE(i, :);
            barIndex = connectedBars(1);
            bar = memberInfo{barIndex};
            angle = 360.00; % Angle for nodes with a single connection
            crossArea = bar.CrossArea;
            
            % Store angles and cross-areas in the structure
            anglesAtNodes(i).Node = i;
            anglesAtNodes(i).Angles = round(angle, 2);
            anglesAtNodes(i).CrossAreas = crossArea;
            
            % Store angle sets
            angleSets{end+1} = strcat(num2str(round(angle, 2)), '-', mat2str(round(crossArea, 2)));
        elseif numConnections > 1
            nodePos = NODE(i, :);
            angles = zeros(1, numConnections);
            crossAreas = zeros(numConnections, 1);
            
            % Calculate angles of each connected bar with the positive x-axis
            for k = 1:numConnections
                barIndex = connectedBars(k);
                bar = memberInfo{barIndex};
                otherNode = bar.Start;
                if isequal(nodePos, bar.Start)
                    otherNode = bar.End;
                end
                vec = otherNode - nodePos;
                angle = atan2d(vec(2), vec(1)); % Angle with the horizontal
                if angle < 0
                    angle = angle + 360; % Ensure the angle is in [0, 360]
                end
                angles(k) = round(angle, 2);
                crossAreas(k) = bar.CrossArea;
            end
            
            % Sort angles and their corresponding cross-areas together
            [angles, sortIdx] = sort(angles, 'ascend');
            crossAreas = crossAreas(sortIdx, :);
            
            % Calculate differences between consecutive angles to get angles between bars
            angleDiffs = diff([angles, angles(1) + 360]);
            
            % Filter out nodes with all angles being 180.00
            if ~all(angleDiffs == 180.00)
                % Store angles and cross-areas in the structure
                anglesAtNodes(i).Node = i;
                anglesAtNodes(i).Angles = round(angleDiffs, 2);
                anglesAtNodes(i).CrossAreas = crossAreas;
                
                % Store angle sets
                angleSetStr = strcat(num2str(round(angleDiffs, 2)), '-', mat2str(round(crossAreas, 2)));
                angleSets{end+1} = angleSetStr;
            end
        end
    end
    
    % Print angles and corresponding cross-areas
    for i = 1:length(anglesAtNodes)
        if ~isempty(anglesAtNodes(i).Angles)
            if anglesAtNodes(i).Angles == 360.00
                fprintf('Node %d: %.2f degrees\n', anglesAtNodes(i).Node, anglesAtNodes(i).Angles);
                fprintf('Node %d (Cross-Area): %.2f\n', anglesAtNodes(i).Node, anglesAtNodes(i).CrossAreas);
            else
                fprintf('Node %d: %.2f degrees', anglesAtNodes(i).Node, anglesAtNodes(i).Angles(1));
                for j = 2:length(anglesAtNodes(i).Angles)
                    fprintf(', %.2f degrees', anglesAtNodes(i).Angles(j));
                end
                fprintf('\n');
                fprintf('Node %d (Cross-Areas): %.2f', anglesAtNodes(i).Node, anglesAtNodes(i).CrossAreas(1));
                for j = 2:length(anglesAtNodes(i).CrossAreas)
                    fprintf(', %.2f', anglesAtNodes(i).CrossAreas(j));
                end
                fprintf('\n');
            end
        end
    end
end

% Function to count occurrences of different angle sets with tolerance
function angleSetCounts = countAngleSetOccurrencesWithTolerance(anglesAtNodes, tolerance)
    angleSetCounts = containers.Map('KeyType', 'char', 'ValueType', 'int32');
    
    function isSimilar = isSimilarAngle(angle1, angle2, tolerance)
        isSimilar = abs(angle1 - angle2) <= tolerance;
    end

    function existingKey = findSimilarKey(map, newKey, tolerance)
        existingKey = '';
        keysList = map.keys();
        newAngles = str2num(newKey); %#ok<ST2NM>
        for i = 1:length(keysList)
            existingAngles = str2num(keysList{i}); %#ok<ST2NM>
            if length(existingAngles) == length(newAngles)
                similar = true;
                for j = 1:length(existingAngles)
                    if ~isSimilarAngle(existingAngles(j), newAngles(j), tolerance)
                        similar = false;
                        break;
                    end
                end
                if similar
                    existingKey = keysList{i};
                    return;
                end
            end
        end
    end

    for i = 1:length(anglesAtNodes)
        if ~isempty(anglesAtNodes(i).Angles)
            % Create a sorted key of angles
            [sortedAngles, ~] = sort(anglesAtNodes(i).Angles);
            angleSetStr = num2str(sortedAngles);

            existingKey = findSimilarKey(angleSetCounts, angleSetStr, tolerance);
            if ~isempty(existingKey)
                angleSetCounts(existingKey) = angleSetCounts(existingKey) + 1;
            else
                angleSetCounts(angleSetStr) = 1;
            end
        end
    end
end

% Function to count occurrences of different angle sets including cross-sectional areas with tolerances
function angleSetCountsIncludingAreas = countAngleSetOccurrencesIncludingAreasWithTolerance(anglesAtNodes, angleTolerance, areaTolerance)
    angleSetCountsIncludingAreas = containers.Map('KeyType', 'char', 'ValueType', 'int32');
    
    function isSimilar = isSimilarValue(value1, value2, tolerance)
        isSimilar = abs(value1 - value2) <= tolerance;
    end

    function existingKey = findSimilarKey(map, newKey, angleTolerance, areaTolerance)
        existingKey = '';
        keysList = map.keys();
        newKeyParts = split(newKey, '-');
        newAngles = str2num(newKeyParts{1}); %#ok<ST2NM>
        newAreas = str2num(newKeyParts{2}); %#ok<ST2NM>
        for i = 1:length(keysList)
            existingKeyParts = split(keysList{i}, '-');
            existingAngles = str2num(existingKeyParts{1}); %#ok<ST2NM>
            existingAreas = str2num(existingKeyParts{2}); %#ok<ST2NM>
            if length(existingAngles) == length(newAngles) && length(existingAreas) == length(newAreas)
                similar = true;
                for j = 1:length(existingAngles)
                    if ~isSimilarValue(existingAngles(j), newAngles(j), angleTolerance)
                        similar = false;
                        break;
                    end
                end
                for j = 1:length(existingAreas)
                    if ~isSimilarValue(existingAreas(j), newAreas(j), areaTolerance)
                        similar = false;
                        break;
                    end
                end
                if similar
                    existingKey = keysList{i};
                    return;
                end
            end
        end
    end

    for i = 1:length(anglesAtNodes)
        if ~isempty(anglesAtNodes(i).Angles)
            % Create a sorted key of angles and cross-areas
            [sortedAngles, sortIdx] = sort(anglesAtNodes(i).Angles);
            sortedCrossAreas = anglesAtNodes(i).CrossAreas(sortIdx, :);
            % Sort cross-areas based on numerical values
            [~, crossAreaSortIdx] = sortrows(sortedCrossAreas);
            sortedCrossAreas = sortedCrossAreas(crossAreaSortIdx, :);
            angleSetStr = strcat(num2str(sortedAngles), '-', mat2str(sortedCrossAreas));

            existingKey = findSimilarKey(angleSetCountsIncludingAreas, angleSetStr, angleTolerance, areaTolerance);
            if ~isempty(existingKey)
                angleSetCountsIncludingAreas(existingKey) = angleSetCountsIncludingAreas(existingKey) + 1;
            else
                angleSetCountsIncludingAreas(angleSetStr) = 1;
            end
        end
    end
end
