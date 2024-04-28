import numpy as np
import matplotlib.pyplot as plt


def plot_contour(x, y, value, title, size, color_interpolation=40):
    """
    画出均匀笛卡尔网格的等高线图
    :param x: 1D x坐标
    :param y: 1D y坐标
    :param value: 2D等高线高度值
    :param title: 图片标题
    :param size: 图片大小
    :param color_interpolation: 云图颜色插值数
    :return: None
    """
    mesh_x, mesh_y = np.meshgrid(x, y)
    plt.figure(figsize=size)
    value = np.flipud(np.rot90(value))

    # 等高线
    c = plt.contour(mesh_x, mesh_y, value, colors='black', linewidths=0.75)
    plt.clabel(c, inline_spacing=20, fmt='%.1f', fontsize=5)
    # 云图
    plt.contourf(mesh_x, mesh_y, value, color_interpolation, cmap='rainbow')
    plt.colorbar()

    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.title(title, fontsize=18)
    plt.show()


def main():
    x_points, y_points = 100, 100
    x = np.arange(x_points) / x_points
    y = np.arange(y_points) / x_points
    value = np.random.normal(0, 1, (x_points, y_points))
    plot_contour(
        x=x,
        y=y,
        value=value,
        title='title',
        size=[8, 6.5]
    )


if __name__ == '__main__':
    main()
