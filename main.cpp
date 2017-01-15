#include <iostream>
#include <vector>
#include <functional>
#include <memory>
#include <cmath>
#include <assert.h>
#include <limits>
#include <random>

typedef std::pair<float, float> Coordinates;

struct Objective {

    typedef std::function<float(Coordinates)> CallbackFunction;
    typedef std::function<Coordinates(Coordinates)> CallbackGradient;

    Objective(CallbackFunction foo, CallbackGradient grad)
            : callable_function(foo), callable_gradient(grad) {}


    float compute(Coordinates c) const {
        return callable_function(c);
    }

    Coordinates gradient(Coordinates c) const {
        return callable_gradient(c);
    }

private:
    CallbackFunction callable_function;
    CallbackGradient callable_gradient;

};


struct Optimizer {
    virtual void minimize(Objective const &) = 0;
};

struct SGD : Optimizer {
    SGD(std::pair<float, float> start_point, size_t steps = 10000, float lr = 0.01)
            : pos(start_point), count_steps(steps), learning_rate(lr) {}

    void minimize(Objective const &loss) {

        float loss_value = std::numeric_limits<float>::max();
        for (size_t t = 0; t < count_steps; ++t) {
            loss_value = loss.compute(pos);
            auto gradient = loss.gradient(pos);

            pos.first -= learning_rate * gradient.first;
            pos.second -= learning_rate * gradient.second;

            update_history.push_back(pos);
        }

        std::cout << "Finished at { x = " << pos.first << ", y = " << pos.second << " }" << std::endl;
        std::cout << "Loss: " << loss_value << std::endl;
    }

private:
    std::pair<float, float> pos;
    size_t count_steps;
    float learning_rate;

    std::vector<std::pair<float, float>> update_history{};
};


struct MomentumOptimizer : Optimizer {

    MomentumOptimizer(Coordinates start_point,
                      size_t steps = 10000,
                      float lr = 0.01,
                      float momentum = 0.9)
            : curr_point(start_point),
              count_steps(steps),
              learning_rate(lr),
              gamma(momentum) {}

    void minimize(Objective const &loss) {

        float loss_value = std::numeric_limits<float>::max();

        Coordinates v_curr;

        // v_0 is undefined. E.g. make it small random numbers
        Coordinates v_prev;

        {
            std::random_device device;
            std::mt19937 mt(device());
            std::uniform_real_distribution<> dist(-0.1, 0.1);
            v_prev = std::make_pair(dist(mt), dist(mt));
        }

        for (size_t t = 0; t < count_steps; ++t) {
            loss_value = loss.compute(curr_point);
            Coordinates gradient = loss.gradient(curr_point);

            // v_t = beta_1 * v_{t-1} + (1 - beta_1) * lr * gradient
            float v_t_x = (float) (gamma * v_prev.first + (1.0 - gamma) * learning_rate * gradient.first);
            float v_t_y = (float) (gamma * v_prev.second + (1.0 - gamma) * learning_rate * gradient.second);
            v_curr = std::make_pair(v_t_x, v_t_y);

            curr_point.first -= v_curr.first;
            curr_point.second -= v_curr.second;

            v_prev = v_curr;

            update_history.push_back(curr_point);
        }

        std::cout << "Finished at { x = " << curr_point.first << ", y = " << curr_point.second << " }" << std::endl;
        std::cout << "Loss: " << loss_value << std::endl;
    }

private:
    std::pair<float, float> curr_point;
    size_t count_steps;
    float learning_rate;
    float gamma;

    std::vector<std::pair<float, float>> update_history{};
};


struct NesterovAcceleratedGradient : Optimizer {

    NesterovAcceleratedGradient(Coordinates start_point,
                                size_t steps = 10000,
                                float lr = 0.01,
                                float momentum = 0.9)
            : pos(start_point),
              count_steps(steps),
              learning_rate(lr),
              gamma(momentum) {}

    void minimize(Objective const &loss) {

        float loss_value = std::numeric_limits<float>::max();

        Coordinates v_curr;

        /* v_0 is undefined. E.g. make it small random numbers */
        Coordinates v_prev;

        {
            std::random_device device;
            std::mt19937 mt(device());
            std::uniform_real_distribution<> dist(-0.1, 0.1);
            v_prev = std::make_pair(dist(mt), dist(mt));
        }

        for (size_t t = 0; t < count_steps; ++t) {

            /* Key improvement w.r.t Momentum optimizer is to stabilize gradient along the future direction */
            auto future_grad_1_3 = loss.gradient(std::make_pair(pos.first - 0.3333 * (gamma * v_prev.first),
                                                                pos.second - 0.3333 * (gamma * v_prev.second)));

            auto future_grad_2_3 = loss.gradient(std::make_pair(pos.first - 0.6666 * (gamma * v_prev.first),
                                                                pos.second - 0.6666 * (gamma * v_prev.second)));

            auto future_grad = loss.gradient(std::make_pair(pos.first - gamma * v_prev.first,
                                                            pos.second - gamma * v_prev.second));


            /* Just mean of gradients along the future direction */
            auto gradient_x = 0.3333 * (future_grad_1_3.first + future_grad_2_3.first + future_grad.first);
            auto gradient_y = 0.3333 * (future_grad_1_3.second + future_grad_2_3.second + future_grad.second);


            // v_t = beta_1 * v_{t-1} + (1 - beta_1) * lr * gradient
            v_curr.first = gamma * v_prev.first + (1 - gamma) * learning_rate * gradient_x;
            v_curr.second = gamma * v_prev.second + (1 - gamma) * learning_rate * gradient_y;

            pos.first -= v_curr.first;
            pos.second -= v_curr.second;
            loss_value = loss.compute(pos);

            v_prev = v_curr;

            update_history.push_back(pos);
        }

        std::cout << "Finished at { x = " << pos.first << ", y = " << pos.second << " }" << std::endl;
        std::cout << "Loss: " << loss_value << std::endl;
    }

private:
    std::pair<float, float> pos;
    size_t count_steps;
    float learning_rate;
    float gamma;

    std::vector<std::pair<float, float>> update_history{};
};


struct Adagrad : Optimizer {

    Adagrad(Coordinates start_point, size_t steps = 10000, float lr = 0.1, float eps = 1e-6)
            : pos(start_point), count_steps(steps), learning_rate(lr), eps(eps) {}

    void minimize(Objective const &loss) {

        float loss_value = std::numeric_limits<float>::max();

        Coordinates G_t;

        for (size_t t = 0; t < count_steps; ++t) {
            loss_value = loss.compute(pos);
            Coordinates gradient = loss.gradient(pos);

            /*  G_{t + 1} = G_t + g^2 */
            G_t.first += (gradient.first * gradient.first);
            G_t.second += (gradient.second * gradient.second);

            pos.first -= learning_rate * gradient.first / sqrt(G_t.first + eps);
            pos.second -= learning_rate * gradient.second / sqrt(G_t.second + eps);
        }

        std::cout << "Finished at { x = " << pos.first << ", y = " << pos.second << " }" << std::endl;
        std::cout << "Loss: " << loss_value << std::endl;
    }


private:
    std::pair<float, float> pos;
    size_t count_steps;
    float learning_rate;
    float eps;
};


struct RMSProp : Optimizer {

    RMSProp(Coordinates start_point, size_t steps = 10000, float lr = 0.01, float gamma = 0.9, float eps = 1e-6)
            : pos(start_point), count_steps(steps), learning_rate(lr), gamma(gamma), eps(eps) {}

    void minimize(Objective const &loss) {

        float loss_value = std::numeric_limits<float>::max();
        Coordinates msg = std::make_pair(0.0, 0.0); // mean square gradient moving average
        for (size_t t = 0; t < count_steps; ++t) {
            loss_value = loss.compute(pos);
            Coordinates gradient = loss.gradient(pos);

            msg.first = gamma * msg.first + (1 - gamma) * (gradient.first * gradient.first);
            msg.second = gamma * msg.second + (1 - gamma) * (gradient.second * gradient.second);

            pos.first -= learning_rate * gradient.first / sqrt(msg.first + eps);
            pos.second -= learning_rate * gradient.second / sqrt(msg.second + eps);
        }

        std::cout << "Finished at { x = " << pos.first << ", y = " << pos.second << " }" << std::endl;
        std::cout << "Loss: " << loss_value << std::endl;
    }


private:
    std::pair<float, float> pos;
    size_t count_steps;
    float learning_rate;
    float gamma;
    float eps;
};

struct Adam : Optimizer {

    Adam(Coordinates start_point, size_t steps = 10000, float lr = 0.01, float b1 = 0.9, float b2 = 0.99999,
         float eps = 1e-8)
            : pos(start_point), count_steps(steps), learning_rate(lr), beta_1(b1), beta_2(b2), eps(eps) {}

    void minimize(Objective const &loss) {

        float loss_value = std::numeric_limits<float>::max();

        Coordinates mg; // mean gradient moving average
        Coordinates m_t; // mean gradient normalized by (1 - beta_1)

        Coordinates msg; // mean square gradient moving average
        Coordinates v_t; // mean square gradient normalized by (1 - beta_2)

        for (size_t t = 0; t < count_steps; ++t) {
            loss_value = loss.compute(pos);
            Coordinates gradient = loss.gradient(pos);

            mg.first = beta_1 * mg.first + (1.0 - beta_1) * gradient.first;
            mg.second = beta_1 * mg.second + (1.0 - beta_1) * gradient.second;
            m_t.first = mg.first / (1.0 - beta_1);
            m_t.second = mg.second / (1.0 - beta_1);


            msg.first = beta_2 * msg.first + (1.0 - beta_2) * (gradient.first * gradient.first);
            msg.second = beta_2 * msg.second + (1.0 - beta_2) * (gradient.second * gradient.second);
            v_t.first = msg.first / (1.0 - beta_2);
            v_t.second = msg.second / (1.0 - beta_2);

            pos.first -= learning_rate * m_t.first / sqrt(v_t.first + eps);
            pos.second -= learning_rate * m_t.second / sqrt(v_t.second + eps);
        }

        std::cout << "Finished at { x = " << pos.first << ", y = " << pos.second << " }" << std::endl;
        std::cout << "Loss: " << loss_value << std::endl;
    }


private:
    std::pair<float, float> pos;
    size_t count_steps;
    float learning_rate;
    float beta_1;
    float beta_2;
    float eps;
};

void run_all_with(Objective const &loss, Coordinates const &start_point, size_t max_steps) {
    auto sgd = std::make_shared<SGD>(start_point);
    sgd->minimize(loss);

    auto momentum = std::make_shared<MomentumOptimizer>(start_point, max_steps);
    momentum->minimize(loss);

    auto nesterov = std::make_shared<NesterovAcceleratedGradient>(start_point, max_steps);
    nesterov->minimize(loss);

    auto adagrad = std::make_shared<Adagrad>(start_point, max_steps);
    adagrad->minimize(loss);

    auto rms = std::make_shared<RMSProp>(start_point, max_steps);
    rms->minimize(loss);

    auto adam = std::make_shared<Adam>(start_point, max_steps);
    adam->minimize(loss);
}

void test_paraboloid() {
    Objective loss(
            [](std::pair<float, float> coord) {
                float val = std::pow(coord.first - 3, 2) + std::pow(coord.second - 3, 2);
                return val;
            },

            [](std::pair<float, float> coord) {
                float dfdx = 2 * (coord.first - 3);
                float dfdy = 2 * (coord.second - 3);
                return std::make_pair(dfdx, dfdy);
            });

    assert(loss.compute(std::make_pair(3, 3)) == 0.0);
    assert(loss.gradient(std::make_pair(3, 3)).first == 0.0);
    assert(loss.gradient(std::make_pair(3, 3)).second == 0.0);

    auto start_point = std::make_pair(100.0, 100.0);


    std::cout << "Parabilic objective :" << std::endl << std::endl;
    run_all_with(loss, start_point, 1000000);
    std::cout << "==================================" << std::endl;
}

void test_rosenbrock() {
    Objective loss(
            [](std::pair<float, float> coord) {
                return std::pow(1 - coord.first, 2) + 100 * std::pow(coord.second - std::pow(coord.first, 2), 2);
            },

            [](std::pair<float, float> coord) {
                float dfdx = 2 * (200 * std::pow(coord.first, 3) - 200 * coord.first * coord.second + coord.first - 1);
                float dfdy = 200 * (coord.second - std::pow(coord.first, 2));
                return std::make_pair(dfdx, dfdy);
            });

    auto start_point = std::make_pair(0.0, 3.0);

    std::cout << "Rosenbrock objective :" << std::endl << std::endl;
    run_all_with(loss, start_point, 1000000);
    std::cout << "==================================" << std::endl;
}

int main() {
    test_paraboloid();
    test_rosenbrock();
    return 0;
}