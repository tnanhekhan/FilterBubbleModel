var XYScaledNetworkModule = function(svg_width, svg_height) {
    // Create the svg tag:
    var svg_tag = "<svg id='XYScaledNetworkModule_d3' width='" + svg_width + "' height='" + svg_height + "' " +
        "style='border:1px dotted;'></svg>";

    // Append svg to #elements:
    const svg_node = d3.create("svg");
    svg_node
        .attr("id", "XYScaledNetworkModule_d3")
        .attr("width", svg_width)
        .attr("height", svg_height);
        
    document.getElementById("elements").appendChild(svg_node.node());

    var svg = d3.select("#XYScaledNetworkModule_d3").style('border', '3px solid black');

    var width = svg_width
    var height = svg_height

    // scaling x, y values
    var xScale = d3.scaleLinear().domain([-1,1]).range([0,width])
    var yScale = d3.scaleLinear().domain([-1,1]).range([height,0])

    var g = svg.append("g")
            .classed("network_root", true).style('pointer-events', 'all')

    var tooltip = d3.select("body").append("div")
        .attr("class", "tooltip")
        .style("opacity", 0)
        .style('pointer-events', 'none')
        .style('background-color', 'white')

    var links = g.append("g")
        .attr("class", "links")
        .style('pointer-events', 'all')

    var nodes = g.append("g")
        .attr("class", "nodes")
        .style('pointer-events', 'all')

    this.render = function(data) {
        var graph = JSON.parse(JSON.stringify(data));
        var hide_friends_choice = document.getElementById("hide_friends_id");
        var hide_itemlinks_choice = document.getElementById("hide_itemlinks_id");

        simulation = d3.forceSimulation()
        .nodes(graph.nodes)
        .force("charge", d3.forceManyBody()
            .strength(-80)
            .distanceMin(2))
        .force("link", d3.forceLink(graph.edges))
        .force("center", d3.forceCenter())
        .stop();

        for (var i = 0, n = Math.ceil(Math.log(simulation.alphaMin()) / Math.log(1 - simulation.alphaDecay())); i < n; ++i) {
            simulation.tick();
        }
        
        if(graph.datacollector_data != null){
            svg
            .data(graph.datacollector_data)
            .style('border', function(d){
                if(parseFloat(d.dist_between_sharers) <= parseFloat(d.dist_between_users_and_items)
                    && parseFloat(d.dist_between_users_and_items) <= parseFloat(d.dist_between_friends)){
                    return '3px solid orange'
                }
                else if(parseFloat(d.dist_between_friends) <= parseFloat(d.dist_between_sharers)
                    && parseFloat(d.dist_between_sharers) <= parseFloat(d.dist_between_users_and_items)){
                    return '3px solid red'
                }
                else if(parseFloat(d.dist_between_users_and_items) <= parseFloat(d.dist_between_sharers)
                    && parseFloat(d.dist_between_sharers) <= parseFloat(d.dist_between_friends)){
                    return '3px solid green'
                }else{
                    return '3px solid black'
                }
            })
        }else{
            svg.style('border', '3px solid black')
        }

        // adding x axis
        var x_axis = d3.axisBottom().scale(xScale)
        var x_axis_translate = height - 5;
        svg.append("g")
            .attr("transform","translate(0," + x_axis_translate + ")")
            .call(x_axis)
            .selectAll("text")
            .attr("transform", "translate(0," + -20 + ")")
            .style("text-anchor", "start");

        // adding y axis
        var y_axis = d3.axisLeft().scale(yScale)
        var y_axis_translate = width;

        svg.append("g")
            .attr("transform","translate(" + y_axis_translate + ",0)")
            .call(y_axis)
        
        links
            .selectAll("line")
            .data(graph.edges)
            .enter()
            .append("line")
            .style("visibility", function(d){
                if(d.id.toString().includes("friend")){
                    if(hide_friends_choice.checked){
                        return "hidden"
                    }else{
                        return "visible"
                    }
                }

                if(d.id.toString().includes("itemlink")){
                    if(hide_itemlinks_choice.checked){
                        return "hidden"
                    }else{
                        return "visible"
                    }
                }
            });

        //Turn on/off rendering friends edges
        hide_friends_choice.addEventListener("click", function(){
            links
                .selectAll("line")
                .data(graph.edges)
                .style("visibility", function(d){
                    if(d.id.toString().includes("friend")){
                        if(hide_friends_choice.checked){
                            return "hidden"
                        }else{
                            return "visible"
                        }
                    }
                });
        });

        //Turn on/off rendering infolink/itemlink edges
        hide_itemlinks_choice.addEventListener("click", function(){
            links
                .selectAll("line")
                .data(graph.edges)
                .style("visibility", function(d){
                    if(d.id.toString().includes("itemlink")){
                        if(hide_itemlinks_choice.checked){
                            return "hidden"
                        }else{
                            return "visible"
                        }
                    }
                });
        });

        links
            .selectAll("line")
            .data(graph.edges)
            .style("stroke-dasharray", ("2, 2"))
            .attr("x1", function(d) { return xScale(d.source.x); })
            .attr("y1", function(d) { return yScale(d.source.y); })
            .attr("x2", function(d) { return xScale(d.target.x); })
            .attr("y2", function(d) { return yScale(d.target.y); })
            .attr("stroke-width", function(d) { return d.width; })
            .attr("stroke", function(d) { return d.color;})
            .style("visibility", function(d){
                if(d.id.toString().includes("friend")){
                    if(hide_friends_choice.checked){
                        return "hidden"
                    }else{
                        return "visible"
                    }
                }

                if(d.id.toString().includes("itemlink")){
                    if(hide_itemlinks_choice.checked){
                        return "hidden"
                    }else{
                        return "visible"
                    }
                }
            });

        links
            .selectAll("line")
            .data(graph.edges)
            .exit()
                .remove();

        nodes
            .selectAll("circle")
            .data(graph.nodes)
            .enter()
            .append("circle")
            .on("mouseover", function(d) {
                tooltip.transition()
                    .duration(200)
                    .style("opacity", .9);
                tooltip.html(d.tooltip)
                    .style("left", (d3.event.pageX) + "px")
                    .style("top", (d3.event.pageY) + "px");
                console.log(d)
            })
            .on("mouseout", function() {
                tooltip.transition()
                    .duration(500)
                    .style("opacity", 0);
            })
            .style("opacity", function(d){
                if(d.color == "#e6acac"){
                    return 0.5
                }
            });

        nodes.selectAll("circle")
            .data(graph.nodes)
            .transition()
            .duration(700)
            .ease(d3.easeLinear)
            .attr("cx", function(d) { return xScale(d.x); })
            .attr("cy", function(d) { return yScale(d.y); })
            .attr("r", function(d) { return d.size; })
            .attr("fill", function(d) { return d.color; })
            .style("opacity", function(d){
                if(d.color == "#e6acac"){
                    return 0.5;
                }
            });

        nodes.selectAll("circle")
            .data(graph.nodes)
            .exit()
                .remove();
    };

    this.reset = function() {

    };
};
