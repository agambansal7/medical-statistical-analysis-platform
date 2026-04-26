declare module 'plotly.js-dist-min' {
  interface PlotData {
    x?: any[];
    y?: any[];
    type?: string;
    mode?: string;
    name?: string;
    marker?: any;
    line?: any;
    fill?: string;
    fillcolor?: string;
    hovertemplate?: string;
    showlegend?: boolean;
    hoverinfo?: string;
    text?: any[];
    textposition?: string;
    [key: string]: any;
  }

  interface Layout {
    title?: any;
    xaxis?: any;
    yaxis?: any;
    width?: number;
    height?: number;
    template?: string;
    paper_bgcolor?: string;
    plot_bgcolor?: string;
    font?: any;
    margin?: any;
    hovermode?: string;
    legend?: any;
    annotations?: any[];
    [key: string]: any;
  }

  interface Config {
    responsive?: boolean;
    displayModeBar?: boolean;
    modeBarButtonsToRemove?: string[];
    displaylogo?: boolean;
    toImageButtonOptions?: {
      format?: string;
      filename?: string;
      height?: number;
      width?: number;
      scale?: number;
    };
    [key: string]: any;
  }

  interface DownloadImageOptions {
    format: string;
    filename: string;
    height?: number;
    width?: number;
    scale?: number;
  }

  interface Plotly {
    newPlot: (
      root: HTMLElement,
      data: PlotData[],
      layout?: Layout,
      config?: Config
    ) => Promise<any>;
    purge: (root: HTMLElement) => void;
    downloadImage: (root: HTMLElement, opts: DownloadImageOptions) => Promise<void>;
    react: (
      root: HTMLElement,
      data: PlotData[],
      layout?: Layout,
      config?: Config
    ) => Promise<any>;
  }

  const plotly: Plotly;
  export default plotly;
}
