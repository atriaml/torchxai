const option = {
  legend: {
    data: ["bar", "bar2", "bar3", "bar4"],
    left: "10%",
  },
  brush: {
    toolbox: ["rect", "polygon", "lineX", "lineY", "keep", "clear"],
    xAxisIndex: 0,
  },
  toolbox: {
    feature: {
      magicType: {
        type: ["stack"],
      },
      dataView: {},
    },
  },
  tooltip: {},
  xAxis: {
    data: (() => {
      let xAxisData = [];
      for (let i = 0; i < 10; i++) {
        xAxisData.push("Class" + i);
      }
      return xAxisData;
    })(),
    name: "X Axis",
    axisLine: { onZero: true },
    splitLine: { show: false },
    splitArea: { show: false },
  },
  yAxis: {},
  grid: {
    bottom: 100,
  },
  series: [
    {
      name: "bar",
      type: "bar",
      stack: "one",
      emphasis: {
        itemStyle: {
          shadowBlur: 10,
          shadowColor: "rgba(0,0,0,0.3)",
        },
      },
      data: (() => {
        let data1 = [];
        for (let i = 0; i < 10; i++) {
          data1.push(+(Math.random() * 2).toFixed(2));
        }
        return data1;
      })(),
    },
    {
      name: "bar2",
      type: "bar",
      stack: "one",
      emphasis: {
        itemStyle: {
          shadowBlur: 10,
          shadowColor: "rgba(0,0,0,0.3)",
        },
      },
      data: (() => {
        let data2 = [];
        for (let i = 0; i < 10; i++) {
          data2.push(+(Math.random() * 5).toFixed(2));
        }
        return data2;
      })(),
    },
    {
      name: "bar3",
      type: "bar",
      stack: "two",
      emphasis: {
        itemStyle: {
          shadowBlur: 10,
          shadowColor: "rgba(0,0,0,0.3)",
        },
      },
      data: (() => {
        let data3 = [];
        for (let i = 0; i < 10; i++) {
          data3.push(+(Math.random() + 0.3).toFixed(2));
        }
        return data3;
      })(),
    },
    {
      name: "bar4",
      type: "bar",
      stack: "two",
      emphasis: {
        itemStyle: {
          shadowBlur: 10,
          shadowColor: "rgba(0,0,0,0.3)",
        },
      },
      data: (() => {
        let data4 = [];
        for (let i = 0; i < 10; i++) {
          data4.push(+Math.random().toFixed(2));
        }
        return data4;
      })(),
    },
  ],
};
